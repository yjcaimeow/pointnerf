import sys
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from models import create_model
from models.mvs import mvs_utils, filter_utils
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
import lpips
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
try:
    from skimage.measure import compare_ssim
    from skimage.measure import compare_psnr
except:
    from skimage.metrics import structural_similarity
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

    def compare_ssim(gt, img, win_size, multichannel=True):
        return structural_similarity(gt, img, win_size=win_size, multichannel=multichannel)

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def get_latents_fn(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)

def nearest_view(campos, raydir, xyz, id_list):
    cam_ind = torch.zeros([0,1], device=campos.device, dtype=torch.long)
    step=10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz),i+step), None, :] - campos[None, ...] # N, M, 3
        dists_norm = torch.norm(dists, dim=-1) # N, M
        dists_dir = dists / (dists_norm[...,None]+1e-6) # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :],dim=-1)) # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1,1)], dim=0) # N, 1
    return cam_ind


def gen_points_filter_embeddings(dataset, visualizer, opt):
    print('-----------------------------------Generate Points-----------------------------------')
    opt.is_train=False
    opt.mode = 1
    model = create_model(opt)
    model.setup(opt)

    model.eval()
    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    intrinsics_full_lst = []
    confidence_filtered_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu= len(dataset.view_id_list) > 300

    imgs_lst, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [],[],[],[],[]
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            model.set_input(data)
            points_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, c2ws, w2cs, intrinsics, near_fars  = model.gen_points()
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())
            intrinsics_all.append(intrinsics_lst[0] if gpu_filter else intrinsics_lst[0])
            extrinsics_all.append(extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy())
            if opt.manual_depth_view !=0:
                confidence_all.append((photometric_confidence_lst[0].cpu() if cpu2gpu else photometric_confidence_lst[0]) if gpu_filter else photometric_confidence_lst[0].cpu().numpy())
            points_mask_all.append((point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0]) if gpu_filter else point_mask_lst[0].cpu().numpy())
            imgs_lst.append(data["images"].cpu())
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            intrinsics_full_lst.append(intrinsics)
            near_fars_all.append(near_fars[0,0])
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            else:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, opt)
        else:
            cam_xyz_all = [cam_xyz_all[i].reshape(-1,3)[points_mask_all[i].reshape(-1),:] for i in range(len(cam_xyz_all))]
            xyz_world_all = [np.matmul(np.concatenate([cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])], axis=-1), np.transpose(np.linalg.inv(extrinsics_all[i][0,...])))[:, :3] for i in range(len(cam_xyz_all))]
            xyz_world_all, cam_xyz_all, confidence_filtered_all = filter_by_masks.range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_all, opt)
            del cam_xyz_all
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape)
        torch.cuda.empty_cache()

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None:
            mask = (xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if opt.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, opt=opt)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)

        if opt.vox_res > 0:
            xyz_world_all, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all)//99999999+1),...].cuda(), opt.vox_res)
            points_vid = points_vid[sampled_pnt_idx,:]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

        xyz_world_all = [xyz_world_all[points_vid[:,0]==i, :] for i in range(len(HDWD_lst))]
        confidence_filtered_all = [confidence_filtered_all[points_vid[:,0]==i] for i in range(len(HDWD_lst))]
        cam_xyz_all = [(torch.cat([xyz_world_all[i], torch.ones_like(xyz_world_all[i][...,0:1])], dim=-1) @ extrinsics_all[i][0].t())[...,:3] for i in range(len(HDWD_lst))]
        points_embedding_all, points_color_all, points_dir_all, points_conf_all = [], [], [], []
        for i in tqdm(range(len(HDWD_lst))):
            if len(xyz_world_all[i]) > 0:
                embedding, color, dir, conf = model.query_embedding(HDWD_lst[i], torch.as_tensor(cam_xyz_all[i][None, ...], device="cuda", dtype=torch.float32), torch.as_tensor(confidence_filtered_all[i][None, :, None], device="cuda", dtype=torch.float32) if len(confidence_filtered_all) > 0 else None, imgs_lst[i].cuda(), c2ws_lst[i], w2cs_lst[i], intrinsics_full_lst[i], 0, pointdir_w=True)
                points_embedding_all.append(embedding)
                points_color_all.append(color)
                points_dir_all.append(dir)
                points_conf_all.append(conf)

        xyz_world_all = torch.cat(xyz_world_all, dim=0)
        points_embedding_all = torch.cat(points_embedding_all, dim=1)
        points_color_all = torch.cat(points_color_all, dim=1) if points_color_all[0] is not None else None
        points_dir_all = torch.cat(points_dir_all, dim=1) if points_dir_all[0] is not None else None
        points_conf_all = torch.cat(points_conf_all, dim=1) if points_conf_all[0] is not None else None

        visualizer.save_neural_points(200, xyz_world_all, points_color_all, data, save_ref=opt.load_points == 0)
        print("vis")
        model.cleanup()
        del model
    return xyz_world_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, [img[0].cpu() for img in imgs_lst], [c2w for c2w in c2ws_lst], [w2c for w2c in w2cs_lst] , intrinsics_all, [list(HDWD) for HDWD in HDWD_lst]

def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst

def render_vid(model, dataset, visualizer, opt, bg_info, steps=0, gen_vid=True):
    print('-----------------------------------Rendering-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step))
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    for i in range(0, total_num):
        data = dataset.get_dummyrot_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        visuals = None
        stime = time.time()

        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            model.set_input(data)

            model.test()
            curr_visuals = model.get_current_visuals(data=data)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    visuals[key][start:end, :] = value.cpu().numpy()

        for key, value in visuals.items():
            visualizer.print_details("{}:{}".format(key, visuals[key].shape))
            visuals[key] = visuals[key].reshape(height, width, 3)
        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i)

    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num), 0)
        print('--------------------------------Finish generating vid--------------------------------')

    return

def test(total_steps, model, dataset, visualizer, opt, bg_info, test_steps=0, gen_vid=False, lpips=True, max_test_psnr=0, \
         max_train_psnr=0, bg_color=None, all_z=None, best_PSNR_half=None, sequence_length_list=None, train_sequence_length_list=None, loss_fn_vgg=None):
    print('-----------------------------------Testing-----------------------------------')
    inference_time = time.time()
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step)) # 1 if test_steps == 10000 else opt.test_num_step

    height = dataset.height*opt.zoom_in_scale
    width = dataset.width*opt.zoom_in_scale
    #visualizer.reset()

    alllist_psnr_train, alllist_psnr_test = [],[]
    alllist_ssim_train, alllist_ssim_test = [],[]
    alllist_lpips_train, alllist_lpips_test = [],[]
    #start, end = 0, 0

    #if opt.neural_render == 'style':
    #    tmp = torch.zeros(size=(1, opt.z_dim))
    #    all_z_new = []
    #    for seq_index in range(len(sequence_length_list)):
    #        start = end
    #        end = end + train_sequence_length_list[seq_index]
    #        seq_codes = all_z[start:end]
    #        seq_codes_filltest = []
    #        train_idx=0
    #        for frame_index in range(sequence_length_list[seq_index]):
    #            if frame_index%10==0:
    #                seq_codes_filltest.append(tmp.squeeze().cpu())
    #            else:
    #                seq_codes_filltest.append(seq_codes[train_idx].cpu())
    #                train_idx = train_idx+1
    #        seq_codes_filltest = torch.stack(seq_codes_filltest)
    #        all_z_new.append(seq_codes_filltest)
    preds, gts = [],[]

    train_psnr_half, test_psnr_half = [],[]
    ssim_train_half, ssim_test_half = [],[]
    lpips_train_half_vgg, lpips_test_half_vgg = [],[]

    seq_frame_index = 0
    seq_index = 0
    for i in range(total_num): # 1 if test_steps == 10000 else opt.test_num_step
        #if seq_frame_index%5 != 0:
        #    seq_frame_index += 1
        #    if seq_frame_index>=sequence_length_list[seq_index]:
        #        seq_frame_index = 0
        #        seq_index = seq_index + 1
        #        alllist_psnr_train.append(train_psnr_half)
        #        alllist_psnr_test.append(test_psnr_half)
        #        alllist_ssim_train.append(ssim_train_half)
        #        alllist_ssim_test.append(ssim_test_half)
        #        alllist_lpips_train.append(lpips_train_half_vgg)
        #        alllist_lpips_test.append(lpips_test_half_vgg)
        #        train_psnr_half, test_psnr_half = [],[]
        #        ssim_train_half, ssim_test_half = [],[]
        #        lpips_train_half_vgg, lpips_test_half_vgg = [],[]
        #    continue
        data = dataset.get_item(i)
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        edge_mask = torch.zeros([height, width], dtype=torch.bool)
        edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
        edge_mask=edge_mask.reshape(-1) > 0
        tmpgts = {}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None
        data.pop('gt_mask', None)

        #if seq_frame_index%10==0:
        #    if seq_frame_index==0:
        #        data['style_code'] = all_z_new[seq_index][1].unsqueeze(0).cuda()
        #    else:
        #        data['style_code'] = ((all_z_new[seq_index][seq_frame_index-1]+all_z_new[seq_index][seq_frame_index+1])/2).unsqueeze(0).cuda()
        #else:
        #    data['style_code'] = all_z_new[seq_index][seq_frame_index].unsqueeze(0).cuda()
        data['bg_color'] = bg_color
        data['train_sequence_length_list'] = train_sequence_length_list
        data['sequence_length_list'] = sequence_length_list
        model.set_input(data)
        output = model.test()

        curr_visuals = model.get_current_visuals(data=data)
        pred = curr_visuals['final_coarse_raycolor']
        gt = tmpgts['gt_image']

        if opt.half_supervision:
            pred_half = pred.reshape(height, width, 3)[height//2:, ...]
            gt_half = gt.reshape(height, width, 3)[height//2:, ...]
        else:
            pred_half = pred.reshape(height, width, 3)
            gt_half = gt.reshape(height, width, 3)

        loss_half = torch.nn.MSELoss().to("cuda")(pred_half.cuda(), gt_half.cuda()).detach()
        psnr_half = mse2psnr(loss_half)

        img_tensor = pred.reshape(height, width, 3)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
        gt_tensor = gt.reshape(height, width, 3)[None].permute(0, 3, 1, 2).float() * 2 - 1.0

        if opt.half_supervision:
            ssim_half = compare_ssim(pred.reshape(height, width, 3).cpu().numpy()[height//2:,...], gt.reshape(height, width, 3).cpu().numpy()[height//2:,...],11, multichannel=True)
            lpips_value_half_vgg = loss_fn_vgg(img_tensor[:,:,height//2:,:], gt_tensor.cuda()[:,:,height//2:,:]).detach().item()
        else:
            ssim_half = compare_ssim(pred.reshape(height, width, 3).cpu().numpy(), gt.reshape(height, width, 3).cpu().numpy(),11, multichannel=True)
            lpips_value_half_vgg = loss_fn_vgg(img_tensor, gt_tensor.cuda()).detach().item()

        if seq_frame_index%10==0:
            test_psnr_half.append(psnr_half.detach().cpu().item())
            ssim_test_half.append(ssim_half)
            lpips_test_half_vgg.append(lpips_value_half_vgg)
        else:
            train_psnr_half.append(psnr_half.detach().cpu().item())
            ssim_train_half.append(ssim_half)
            lpips_train_half_vgg.append(lpips_value_half_vgg)

        preds.append(np.asarray(pred.detach().squeeze().cpu().reshape(height, width, 3)))
        gts.append(np.asarray(gt.detach().squeeze().cpu().reshape(height, width,3)))
        rootdir = os.path.join(opt.checkpoints_dir, 'results')

        seq_frame_index += 1
        if seq_frame_index>=sequence_length_list[seq_index]:
            seq_frame_index = 0
            seq_index = seq_index + 1

            alllist_psnr_train.append(train_psnr_half)
            alllist_psnr_test.append(test_psnr_half)
            alllist_ssim_train.append(ssim_train_half)
            alllist_ssim_test.append(ssim_test_half)
            alllist_lpips_train.append(lpips_train_half_vgg)
            alllist_lpips_test.append(lpips_test_half_vgg)

            train_psnr_half, test_psnr_half = [],[]
            ssim_train_half, ssim_test_half = [],[]
            lpips_train_half_vgg, lpips_test_half_vgg = [],[]

    psnr_train_list, pnsr_test_list = [],[]
    ssim_train_list, ssim_test_list = [],[]
    lpips_train_list, lpips_test_list = [],[]

    for seq_id in range(len(sequence_length_list)):
        test_psnr_half, train_psnr_half = alllist_psnr_test[seq_id], alllist_psnr_train[seq_id]
        ssim_test_half, ssim_train_half = alllist_ssim_test[seq_id], alllist_ssim_train[seq_id]
        lpips_test_half_vgg, lpips_train_half_vgg = alllist_lpips_test[seq_id], alllist_lpips_train[seq_id]

        test_psnr_value_half = (sum(test_psnr_half)/len(test_psnr_half))
        train_psnr_value_half =  (sum(train_psnr_half)/max(1,len(train_psnr_half)))

        test_ssim_value_half = (sum(ssim_test_half)/len(ssim_test_half))
        train_ssim_value_half =  sum(ssim_train_half)/max(1,len(ssim_train_half))

        test_lpips_value_half_vgg = (sum(lpips_test_half_vgg)/len(lpips_test_half_vgg))
        train_lpips_value_half_vgg =  (sum(lpips_train_half_vgg)/max(1,len(lpips_train_half_vgg)))

        psnr_train_list.append(train_psnr_value_half)
        pnsr_test_list.append(test_psnr_value_half)
        ssim_train_list.append(train_ssim_value_half)
        ssim_test_list.append(test_ssim_value_half)
        lpips_train_list.append(train_lpips_value_half_vgg)
        lpips_test_list.append(test_lpips_value_half_vgg)

    psnr_avg_value = (sum(pnsr_test_list)/len(pnsr_test_list))

    if total_steps>0 and best_PSNR_half < psnr_avg_value:
        if os.path.exists(rootdir)==False:
            os.makedirs(rootdir, exist_ok=True)
            for img_index, img in enumerate(gts):
                filepath = os.path.join(rootdir, str(img_index).zfill(4)+'_gt.png')
                if opt.half_supervision:
                    img = img[height//2:,:]
                save_image(np.asarray(img), filepath)
        for img_index, img in enumerate(preds):
            filepath = os.path.join(rootdir, str(img_index).zfill(4)+'_pred.png')
            if opt.half_supervision:
                img = img[height//2:,:]
            save_image(np.asarray(img), filepath)
    return psnr_train_list, pnsr_test_list, ssim_train_list, ssim_test_list, lpips_train_list, lpips_test_list

def update_pcds(model, data_loader, all_z, opt, opacity_thresh):
    print('-----------------------------------Probing Holes-----------------------------------')
    add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_conf = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)
    for i, data in enumerate(data_loader):
        data['style_code'] = all_z[data['id']]
        model.set_input(data)
        output = model.test()
        opacity_flag = output["nerf_opacity"] >= opacity_thresh
        opacity_cum = torch.cumsum(opacity_flag, dim=-1)
        #opacity_mask_tensor = ( opacity_flag * opacity_cum * (opacity_cum<=opt.SR))
        opacity_mask_tensor = ( opacity_flag * opacity_cum * (opacity_cum<=6))
        opacity_mask_tensor = (opacity_mask_tensor>0)[..., None]
        add_pcd_num = torch.sum(opacity_mask_tensor)
#        print (torch.max(output["nerf_opacity"]), '$$$$$$ max opacity')
        if add_pcd_num>0:
            nerf_xyz = torch.masked_select(output["nerf_xyz"], opacity_mask_tensor).view(add_pcd_num, 3).detach().clone()
            nerf_feature = torch.masked_select(output["nerf_feature"], opacity_mask_tensor).view(add_pcd_num, opt.point_features_dim).detach().clone()
            nerf_confidance = torch.masked_select(output["nerf_confidance"], opacity_mask_tensor).view(add_pcd_num, 1).detach().clone()

            add_xyz = torch.cat([add_xyz, nerf_xyz], dim=0)
            add_conf = torch.cat([add_conf, nerf_confidance], dim=0)
            add_embedding = torch.cat([add_embedding, nerf_feature], dim=0)

    return add_xyz, add_embedding, add_conf

def probe_hole(model, dataset, visualizer, opt, bg_info, test_steps=0, opacity_thresh=0.7):
    print('-----------------------------------Probing Holes-----------------------------------')
    add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_conf = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_color = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_dir = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)
    kernel_size = model.opt.kernel_size
    if opt.prob_kernel_size is not None:
        tier = np.sum(np.asarray(opt.prob_tiers) < test_steps)
        print("cal by tier", tier)
        model.opt.query_size = np.asarray(opt.prob_kernel_size[tier*3:tier*3+3])
        print("prob query size =", model.opt.query_size)
    model.opt.prob = 1
    total_num = len(model.top_ray_miss_ids) -1 if opt.prob_mode == 0 and opt.prob_num_step > 1 else len(dataset)

    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    height = dataset.height
    width = dataset.width
    visualizer.reset()

    max_num = len(dataset) // opt.prob_num_step
    take_top = False
    if opt.prob_top == 1 and opt.prob_mode <= 0: # and opt.far_thresh <= 0:
        if getattr(model, "top_ray_miss_ids", None) is not None:
            mask = model.top_ray_miss_loss[:-1] > 0.0
            frame_ids = model.top_ray_miss_ids[:-1][mask][:max_num]
            print(len(frame_ids), max_num)
            print("prob frame top_ray_miss_loss:", model.top_ray_miss_loss)
            take_top = True
        else:
            print("model has no top_ray_miss_ids")
    else:
        frame_ids = list(range(len(dataset)))[:max_num]
        random.shuffle(frame_ids)
        frame_ids = frame_ids[:max_num]
    print("{}/{} has holes, id_lst to prune".format(len(frame_ids), total_num), frame_ids, opt.prob_num_step)
    print("take top:", take_top, "; prob frame ids:", frame_ids)
    with tqdm(range(len(frame_ids))) as pbar:
        for j in pbar:
            i = frame_ids[j]
            pbar.set_description("Processing frame id %d" % i)
            data = dataset.get_item(i)
            bg = data['bg_color'][None, :].cuda()
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([height, width], dtype=torch.bool, device='cuda')
            edge_mask[pixel_idx[0, ..., 1].to(torch.long), pixel_idx[0, ..., 0].to(torch.long)] = 1
            edge_mask = edge_mask.reshape(-1) > 0
            totalpixel = pixel_idx.shape[1]
            gt_image_full = data['gt_image'].cuda()

            probe_keys = ["coarse_raycolor", "ray_mask", "ray_max_sample_loc_w", "ray_max_far_dist", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", "shading_avg_embedding"]
            prob_maps = {}
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)
                output = model.test()
                chunk_pixel_id = data["pixel_idx"].to(torch.long)
                output["ray_mask"] = output["ray_mask"][..., None]

                for key in probe_keys:
                    if "ray_max_shading_opacity" not in output and key != 'coarse_raycolor':
                        break
                    if output[key] is None:
                        prob_maps[key] = None
                    else:
                        if key not in prob_maps.keys():
                            C = output[key].shape[-1]
                            prob_maps[key] = torch.zeros((height, width, C), device="cuda", dtype=output[key].dtype)
                        prob_maps[key][chunk_pixel_id[0, ..., 1], chunk_pixel_id[0, ..., 0], :] = output[key]

            gt_image = torch.zeros((height * width, 3), dtype=torch.float32, device=prob_maps["ray_mask"].device)
            gt_image[edge_mask, :] = gt_image_full
            gt_image = gt_image.reshape(height, width, 3)
            miss_ray_mask = (prob_maps["ray_mask"] < 1) * (torch.norm(gt_image - bg, dim=-1, keepdim=True) > 0.002)
            miss_ray_inds = (edge_mask.reshape(height, width, 1) * miss_ray_mask).squeeze(-1).nonzero() # N, 2

            neighbor_inds = bloat_inds(miss_ray_inds, 1, height, width)
            neighboring_miss_mask = torch.zeros_like(gt_image[..., 0])
            neighboring_miss_mask[neighbor_inds[..., 0], neighbor_inds[...,1]] = 1
            if opt.far_thresh > 0:
                far_ray_mask = (prob_maps["ray_mask"] > 0) * (prob_maps["ray_max_far_dist"] > opt.far_thresh) * (torch.norm(gt_image - prob_maps["coarse_raycolor"], dim=-1, keepdim=True) < 0.1)
                neighboring_miss_mask += far_ray_mask.squeeze(-1)
            neighboring_miss_mask = (prob_maps["ray_mask"].squeeze(-1) > 0) * neighboring_miss_mask * (prob_maps["ray_max_shading_opacity"].squeeze(-1) > opacity_thresh) > 0


            add_xyz = torch.cat([add_xyz, prob_maps["ray_max_sample_loc_w"][neighboring_miss_mask]], dim=0)
            add_conf = torch.cat([add_conf, prob_maps["shading_avg_conf"][neighboring_miss_mask]], dim=0) * opt.prob_mul if prob_maps["shading_avg_conf"] is not None else None
            add_color = torch.cat([add_color, prob_maps["shading_avg_color"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_color"] is not None else None
            add_dir = torch.cat([add_dir, prob_maps["shading_avg_dir"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_dir"] is not None else None
            add_embedding = torch.cat([add_embedding, prob_maps["shading_avg_embedding"][neighboring_miss_mask]], dim=0)

            if len(add_xyz) > -1:
                output = prob_maps["coarse_raycolor"].permute(2,0,1)[None, None,...]
                visualizer.save_ref_views({"images": output}, i, subdir="prob_img_{:04d}".format(test_steps))
    model.opt.kernel_size = kernel_size
    if opt.bgmodel.startswith("planepoints"):
        mask = dataset.filter_plane(add_xyz)
        first_lst, _ = masking(mask, [add_xyz, add_embedding, add_color, add_dir, add_conf], [])
        add_xyz, add_embedding, add_color, add_dir, add_conf = first_lst
    if len(add_xyz) > 0:
        visualizer.save_neural_points("prob{:04d}".format(test_steps), add_xyz, None, None, save_ref=False)
        visualizer.print_details("vis added points to probe folder")
    if opt.prob_mode == 0 and opt.prob_num_step > 1:
        model.reset_ray_miss_ranking()
    del visualizer, prob_maps
    model.opt.prob = 0

    return add_xyz, add_embedding, add_color, add_dir, add_conf

def bloat_inds(inds, shift, height, width):
    inds = inds[:,None,:]
    sx, sy = torch.meshgrid(torch.arange(-shift, shift+1, dtype=torch.long), torch.arange(-shift, shift+1, dtype=torch.long))
    shift_inds = torch.stack([sx, sy],dim=-1).reshape(1, -1, 2).cuda()
    inds = inds + shift_inds
    inds = inds.reshape(-1, 2)
    inds[...,0] = torch.clamp(inds[...,0], min=0, max=height-1)
    inds[...,1] = torch.clamp(inds[...,1], min=0, max=width-1)
    return inds

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def create_all_bg(dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, dummy=False):
    total_num = dataset.total
    height = dataset.height
    width = dataset.width
    bg_ray_lst = []
    random_sample = dataset.opt.random_sample
    for i in range(0, total_num):
        dataset.opt.random_sample = "no_crop"
        if dummy:
            data = dataset.get_dummyrot_item(i)
        else:
            data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        # print("data['pixel_idx']",data['pixel_idx'].shape) # 1, 512, 640, 2
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        start=0
        end = height * width

        data['raydir'] = raydir[:, start:end, :]
        data["pixel_idx"] = pixel_idx[:, start:end, :]
        model.set_input(data)

        xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
        bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"])
        bg_ray = bg_ray.reshape(bg_ray.shape[0], height, width, 3) # 1, 512, 640, 3
        bg_ray_lst.append(bg_ray)
    dataset.opt.random_sample = random_sample
    return bg_ray_lst

def main():
    torch.backends.cudnn.benchmark = True
    from options import TrainOptions
    opt = TrainOptions().parse()
    basedir = "/mnt/lustre/caiyingjie/pointnerf/"
#    basedir = "/home/xschen/yjcai/pointnerf"

    if opt.ddp_train:
        from engine.engine import Engine
        engine = Engine(args=opt)
        seed = engine.local_rank
        torch.manual_seed(seed)
        cudnn.benchmark = True
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank==0:
        writer = SummaryWriter(os.path.join(basedir, 'summaries', opt.checkpoints_dir.split('/')[-1]))

    from data.waymo_ft_dataset_multiseq import WaymoFtDataset
    train_dataset = WaymoFtDataset(opt)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.ddp_train else None

    data_loader = torch.utils.data.DataLoader(train_dataset, \
        sampler=sampler, \
        shuffle=(sampler is None), \
        batch_size=opt.batch_size, \
        num_workers=int(opt.n_threads))
    dataset_size = len(data_loader)
    if local_rank==0:
        print (fmt.RED+'========')
        print (dataset_size)
        print (train_dataset.poses.shape)
        print (train_dataset.intrinsic)
        print (train_dataset.images.shape)
        print ('========'+fmt.END)
    visualizer = Visualizer(opt)
    best_PSNR, best_PSNR_half=0.0,0.0
    best_SSIM, best_SSIM_half=-100.0, -100.0
    best_LPIPS_VGG, best_LPIPS_half_VGG=1.0, 1.0
    best_epoch=0
    opt.vox_res = 400
    with torch.no_grad():
        opt.mode = 2
        print (fmt.RED+'========')
        print ('VOXLIZED POINT CLOUD')
        points_xyz_all_list, points_embedding_all, points_color_all, points_dir_all, points_conf_all = [],[],[],[],[]
        for seq_index, points_xyz_all in enumerate(train_dataset.points_xyz_all):
            points_xyz_all = [points_xyz_all] if not isinstance(points_xyz_all, list) else points_xyz_all
            points_xyz_holder = torch.zeros([0,3], dtype=torch.float32).cpu()
            for i in range(len(points_xyz_all)):
                points_xyz = points_xyz_all[i]
                vox_res = opt.vox_res // (1.5**i)
                _, _, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(points_xyz.cuda() if len(points_xyz) < 80000000 else points_xyz[::(len(points_xyz) // 80000000 + 1), ...].cuda(), vox_res)
                points_xyz = points_xyz[sampled_pnt_idx, :]
                points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)

            print (points_xyz_holder.shape, ':D ----- AFTER VOXLIZED')
            print ('========'+fmt.END)

            points_xyz_all_list.append(points_xyz_holder.cuda())
            points_embedding_all.append(torch.randn((1, len(points_xyz_holder), 32)).cuda())
            points_conf_all.append(torch.ones((1, len(points_xyz_holder), 1)).cuda())

            if "1" in list(opt.point_color_mode):
                points_color_all.append(torch.randn((1, len(points_xyz_holder), 3)).cuda())
                points_dir_all.append(torch.randn((1, len(points_xyz_holder), 3)).cuda())
        all_z = nn.Parameter(get_latents_fn(train_dataset.total, 8, opt.z_dim, device='cuda')[0][0])

        opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
        opt.is_train = True
        opt.mode = 2
        model = create_model(opt)

        bg_color = nn.Parameter(torch.zeros((1, opt.shading_color_channel_num))).cuda()

        if "1" in list(opt.point_color_mode):
            model.set_points(points_xyz_all_list, points_embedding_all, points_color=points_color_all, points_dir=points_dir_all, points_conf=points_conf_all,
                             Rw2c=None, bg_color=bg_color, stylecode=all_z)
        else:
            model.set_points(points_xyz_all_list, points_embedding_all, points_color=None, points_dir=None, points_conf=points_conf_all,
                             Rw2c=None, bg_color=bg_color, stylecode=all_z)
        epoch_count = 1
        total_steps = 0
        del points_xyz_all_list, points_embedding_all, points_color_all, points_dir_all, points_conf_all, all_z
    model.setup(opt, train_len=train_dataset.total)
    model.train()
    if opt.ddp_train:
        model.set_ddp()
    if opt.resume_dir:
        load_path = os.path.join(opt.resume_dir, str(opt.resume_iter)+'_states.pth')
        if os.path.isfile(load_path):
            print ('LOADING HISTORY TOTAL STEPS!')
            state_info = torch.load(load_path, map_location='cpu')
            total_steps = state_info['total_steps']

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"
    test_dataset = WaymoFtDataset(test_opt)

    loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1').cuda()
    if opt.only_test:
        with torch.no_grad():
            psnr_train_list, pnsr_test_list, ssim_train_list, ssim_test_list, lpips_train_list, lpips_test_list = \
                test(epoch, model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=True, bg_color=bg_color, best_PSNR_half=best_PSNR_half, \
                sequence_length_list=test_dataset.sequence_length_list, train_sequence_length_list=train_dataset.sequence_length_list, loss_fn_vgg=loss_fn_vgg)
        exit()
    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    if total_steps > 0:
        for scheduler in model.schedulers:
            for i in range(total_steps):
                scheduler.step()

    test_bg_info = None

    if total_steps == 0 and (train_dataset.total > 30):
        other_states = {
            'epoch_count': 0,
            'total_steps': total_steps,
        }
        model.save_networks(total_steps, other_states)
        visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, 0, total_steps))

    for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
        if opt.ddp_train:
            sampler.set_epoch(epoch)
        for i, data in enumerate(data_loader):
            if opt.maximum_step is not None and total_steps >= opt.maximum_step:
                break
            total_steps += 1
            data['bg_color'] = bg_color
            model.set_input(data)
            model.optimize_parameters(total_steps=total_steps)
            losses = model.get_current_losses()
#            if local_rank==0:
#                for key in losses.keys():
#                    if key != "conf_coefficient":
#                        writer.add_scalar(key, losses[key].item(), total_steps)

            visualizer.accumulate_losses(losses)

            if opt.lr_policy.startswith("iter"):
                model.update_learning_rate(opt=opt, total_steps=total_steps)

            if total_steps and total_steps % opt.print_freq == 0 and local_rank==0:
                visualizer.print_losses(total_steps, epoch, writer)
                visualizer.reset()
                model.print_lr(opt=opt, total_steps=total_steps)

            try:
                if (total_steps % opt.save_iter_freq == 0 and total_steps) or total_steps==1:
                    other_states = {
                        "best_PSNR": best_PSNR,
                        "best_epoch": best_epoch,
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                    }
                    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    model.save_networks(total_steps, other_states)

            except Exception as e:
                visualizer.print_details(e)
            #### test model
            if total_steps % opt.test_freq == 0 or total_steps==1:
                model.opt.is_train = 0
                model.opt.no_loss = 1
                #model.print_lr(opt=opt, total_steps=total_steps)
                with torch.no_grad():
                    psnr_train_list, pnsr_test_list, ssim_train_list, ssim_test_list, lpips_train_list, lpips_test_list = \
                    test(epoch, model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=True, bg_color=bg_color, best_PSNR_half=best_PSNR_half, \
                        sequence_length_list=test_dataset.sequence_length_list, train_sequence_length_list=train_dataset.sequence_length_list, loss_fn_vgg=loss_fn_vgg)
                model.opt.no_loss = 0
                model.opt.is_train = 1

                model.train()

                if local_rank==0:
                    test_psnr_half = (sum(pnsr_test_list)/len(pnsr_test_list))
                    train_psnr_half = (sum(psnr_train_list)/len(psnr_train_list))
                    train_ssim_value_half = (sum(ssim_train_list)/len(ssim_train_list))
                    test_ssim_value_half =(sum(ssim_test_list)/len(ssim_test_list))
                    train_lpips_value_half_vgg = (sum(lpips_train_list)/len(lpips_train_list))
                    test_lpips_value_half_vgg = (sum(lpips_test_list)/len(lpips_test_list))

                    best_epoch = epoch if test_psnr_half > best_PSNR_half else best_epoch
                    best_PSNR = max(train_psnr_half, best_PSNR)
                    best_PSNR_half = max(test_psnr_half, best_PSNR_half)

                    best_SSIM = max(train_ssim_value_half, best_SSIM)
                    best_SSIM_half = max(test_ssim_value_half, best_SSIM_half)

                    best_LPIPS_VGG = min(train_lpips_value_half_vgg, best_LPIPS_VGG)
                    best_LPIPS_half_VGG = min(test_lpips_value_half_vgg, best_LPIPS_half_VGG)

                    writer.add_scalar('PSNR_test', test_psnr_half, total_steps)
                    writer.add_scalar('PSNR_train', train_psnr_half, total_steps)

                    writer.add_scalar('SSIM_test', test_ssim_value_half, total_steps)
                    writer.add_scalar('SSIM_train', train_ssim_value_half, total_steps)

                    writer.add_scalar('LPIPS_test', test_lpips_value_half_vgg, total_steps)
                    writer.add_scalar('LPIPS_train', train_lpips_value_half_vgg, total_steps)

                    print (fmt.GREEN+'========EVALUTION=====')
                    print (opt.checkpoints_dir)
                    visualizer.print_details(f"test at epoch {epoch}")

                    visualizer.print_details(f"===== PSNR =====")
                    visualizer.print_details(f"HALF : train & test: {train_psnr_half}, {test_psnr_half}")
                    visualizer.print_details(f"BEST : {best_PSNR}, {best_PSNR_half} {best_epoch}")

                    visualizer.print_details(f"===== SSIM =====")
                    visualizer.print_details(f"HALF : train & test: {train_ssim_value_half}, {test_ssim_value_half}")
                    visualizer.print_details(f"BEST : {best_SSIM}, {best_SSIM_half} {best_epoch}")

                    visualizer.print_details(f"===== LPIPS VGG =====")
                    visualizer.print_details(f"HALF : train & test: {train_lpips_value_half_vgg}, {test_lpips_value_half_vgg}")
                    visualizer.print_details(f"BEST : {best_LPIPS_VGG}, {best_LPIPS_half_VGG} {best_epoch}")
                    for seq_id in range(len(train_dataset.filenames)):
                        visualizer.print_details(f"====== {train_dataset.filenames[seq_id]} ======")
                        visualizer.print_details(f"psnr          train & test : {psnr_train_list[seq_id]}, {pnsr_test_list[seq_id]}")
                        visualizer.print_details(f"ssim          train & test : {ssim_train_list[seq_id]}, {ssim_test_list[seq_id]}")
                        visualizer.print_details(f"lpips         train & test : {lpips_train_list[seq_id]}, {lpips_test_list[seq_id]}")
                    print (fmt.END)

    writer.close()
    del train_dataset
    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
    model.save_networks(epoch, other_states)

    torch.cuda.empty_cache()
    model.opt.no_loss = 1
    model.opt.is_train = 0

    visualizer.print_details("full datasets test:")
    with torch.no_grad():
        test_lpips_value_half_vgg, train_lpips_value_half_vgg = test(epoch, model, test_dataset, Visualizer(test_opt), \
                    test_opt, test_bg_info, test_steps=total_steps, lpips=True, bg_color=bg_color, best_PSNR_half=best_PSNR_half, \
                    sequence_length_list=test_dataset.sequence_length_list, train_sequence_length_list=train_dataset.sequence_length_list)
    best_epoch = epoch if test_psnr_half > best_PSNR_half else best_epoch
    best_PSNR = max(test_psnr, best_PSNR)
    visualizer.print_details(
        f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_epoch: {best_epoch}")

def save_points_conf(visualizer, xyz, points_color, points_conf, total_steps):
    print("total:", xyz.shape, points_color.shape, points_conf.shape)
    colors, confs = points_color[0], points_conf[0,...,0]
    pre = -1000
    for i in range(12):
        thresh = (i * 0.1) if i <= 10 else 1000
        mask = ((confs <= thresh) * (confs > pre)) > 0
        thresh_xyz = xyz[mask, :]
        thresh_color = colors[mask, :]
        visualizer.save_neural_points(f"{total_steps}-{thresh}", thresh_xyz, thresh_color[None, ...], None, save_ref=False)
        pre = thresh
    exit()

def create_render_dataset(test_opt, opt, total_steps, test_num_step=1):
    test_opt.nerf_splits = ["render"]
    test_opt.split = "render"
    test_opt.name = opt.name + "/vid_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_opt.random_sample_size = 30
    test_dataset = create_dataset(test_opt)
    return test_dataset

def create_test_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.name = opt.name + "/test_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset

def create_comb_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["comb"]
    test_opt.split = "comb"
    test_opt.name = opt.name + "/comb_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
