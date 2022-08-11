import os, sys
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs import mvs_utils, filter_utils
from pprint import pprint
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
torch.manual_seed(0)
np.random.seed(0)
from cprint import *
import random
import cv2
from PIL import Image
from tqdm import tqdm
from utils.util import add_flour, init_distributed_mode
import gc
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)


def nearest_view(campos, raydir, xyz):
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
            # intrinsics    1, 3, 3, 3
            points_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, c2ws, w2cs, intrinsics, near_fars  = model.gen_points()
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)
            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())
            # intrinsics_lst[0] 1, 3, 3
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
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            else:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, opt)
            # print(xyz_ref_lst[0].shape) # 224909, 3
        else:
            cam_xyz_all = [cam_xyz_all[i].reshape(-1,3)[points_mask_all[i].reshape(-1),:] for i in range(len(cam_xyz_all))]
            xyz_world_all = [np.matmul(np.concatenate([cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])], axis=-1), np.transpose(np.linalg.inv(extrinsics_all[i][0,...])))[:, :3] for i in range(len(cam_xyz_all))]
            xyz_world_all, cam_xyz_all, confidence_filtered_all = filter_by_masks.range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_all, opt)
            del cam_xyz_all
        # for i in range(len(xyz_world_all)):
        #     visualizer.save_neural_points(i, torch.as_tensor(xyz_world_all[i], device="cuda", dtype=torch.float32), None, data, save_ref=opt.load_points==0)
        # exit()
        # xyz_world_all = xyz_world_all.cuda()
        # confidence_filtered_all = confidence_filtered_all.cuda()
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape)
        torch.cuda.empty_cache()
        # visualizer.save_neural_points(0, xyz_world_all, None, None, save_ref=False)
        # print("vis 0")

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None:
            mask = (xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if opt.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, opt=opt)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=opt.load_points == 0)
        # print("vis 100")

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

def test(model, data_loader, visualizer, opt, bg_info, test_steps=0, gen_vid=False, lpips=False, writer=None, epoch=None, height=480, width=640):
    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = len(data_loader)
    print("test set size {}, interval {}".format(total_num, opt.test_num_step)) # 1 if test_steps == 10000 else opt.test_num_step
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    opt.name = opt.name + "/test_{}_{}".format(epoch, test_steps)
    visualizer = Visualizer(opt)
    visualizer.reset()
    count = 0
    psnr_list = [[] for x in range(len(opt.scans))]
    psnr_list_ray_masked = [[] for x in range(len(opt.scans))]
    vids, seq_ids = [],[]
    for i, data in enumerate(data_loader):
        raydir = data['raydir'].clone()
        seq_id = data['seq_id'].item()
        vid = data['vid'].item()
        vids.append(vid)
        seq_ids.append(seq_id)
        local_raydir = data['local_raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        if opt.progressive_distill:
            ray_valid_loaded = data['ray_valid_loaded']
            decoded_features_loaded = data['decoded_features_loaded']
            sample_loc_loaded  = data['sample_loc_loaded']
            sample_loc_w_loaded= data['sample_loc_w_loaded']

        edge_mask = torch.zeros([height, width], dtype=torch.bool)
        edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
        edge_mask=edge_mask.reshape(-1) > 0
        np_edge_mask=edge_mask.numpy().astype(bool)
        totalpixel = pixel_idx.shape[1]
        tmpgts = {}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None

        data.pop('gt_mask', None)

        visuals = None
        stime = time.time()
        ray_masks = []
        for k in range(0, totalpixel, chunk_size):
            start = k
            end = min([k + chunk_size, totalpixel])
            data['raydir'] = raydir[:, start:end, :]
            data['local_raydir'] = local_raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            if opt.progressive_distill:
                data['ray_valid_loaded'] = ray_valid_loaded[:, start:end, ...]
                data['decoded_features_loaded'] = decoded_features_loaded[:, start:end, ...]
                data['sample_loc_loaded'] = sample_loc_loaded[:, start:end, ...]
                data['sample_loc_w_loaded']=sample_loc_w_loaded[:,start:end,...]

            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals(data=data)

            chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height, width, 3)).astype(chunk.dtype)
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
            if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                ray_masks.append(model.output["ray_mask"] > 0)
        if len(ray_masks) > 0:
            ray_masks = torch.cat(ray_masks, dim=1)
        pixel_idx=pixel_idx.to(torch.long)
        gt_image = torch.zeros((height*width, 3), dtype=torch.float32)
        gt_image[edge_mask, :] = tmpgts['gt_image'].clone()
        if 'gt_image' in model.visual_names:
            visuals['gt_image'] = gt_image
        if 'gt_mask' in curr_visuals:
            visuals['gt_mask'] = np.zeros((height, width, 3)).astype(chunk.dtype)
            visuals['gt_mask'][np_edge_mask,:] = tmpgts['gt_mask']
        if 'ray_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            visuals['ray_masked_coarse_raycolor'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'ray_depth_masked_gt_image' in model.visual_names:
            visuals['ray_depth_masked_gt_image'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['ray_depth_masked_gt_image'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'gt_image_ray_masked' in model.visual_names:
            visuals['gt_image_ray_masked'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['gt_image_ray_masked'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        for key, value in visuals.items():
            if key in opt.visual_items:
                #visualizer.print_details("{}:{}".format(key, visuals[key].shape))
                visuals[key] = visuals[key].reshape(height, width, 3)

        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, seq_id, vid, opt=opt)

        acc_dict = {}
        if "coarse_raycolor" in opt.test_color_loss_items:
            loss = torch.nn.MSELoss().to("cuda")(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), gt_image.view(1, -1, 3).cuda())
            acc_dict.update({"coarse_raycolor": loss})
            psnr_list[seq_id].append(mse2psnr(loss).item())

        if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
            masked_gt = tmpgts["gt_image"].view(1, -1, 3).cuda()[ray_masks,:].reshape(1, -1, 3)
            ray_masked_coarse_raycolor = torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3)[:,edge_mask,:][ray_masks,:].reshape(1, -1, 3)

            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_gt")
            # filepath = os.path.join("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # tmpgtssave = tmpgts["gt_image"].view(1, -1, 3).clone()
            # tmpgtssave[~ray_masks,:] = 1.0
            # img = np.array(tmpgtssave.view(height,width,3))
            # save_image(img, filepath)
            #
            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_coarse_raycolor")
            # filepath = os.path.join(
            #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # csave = torch.zeros_like(tmpgts["gt_image"].view(1, -1, 3))
            # csave[~ray_masks, :] = 1.0
            # csave[ray_masks, :] = torch.as_tensor(visuals["coarse_raycolor"]).view(1, -1, 3)[ray_masks,:]
            # img = np.array(csave.view(height, width, 3))
            # save_image(img, filepath)
            loss = torch.nn.MSELoss().to("cuda")(ray_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_masked_coarse_raycolor", loss, mse2psnr(loss)))
            psnr_list_ray_masked[seq_id].append(mse2psnr(loss).item())

        if "ray_depth_mask" in model.output and "ray_depth_masked_coarse_raycolor" in opt.test_color_loss_items:
            ray_depth_masks = model.output["ray_depth_mask"].reshape(model.output["ray_depth_mask"].shape[0], -1)
            masked_gt = torch.masked_select(tmpgts["gt_image"].view(1, -1, 3).cuda(), (ray_depth_masks[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
            ray_depth_masked_coarse_raycolor = torch.masked_select(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), ray_depth_masks[..., None].expand(-1, -1, 3).reshape(1, -1, 3))

            loss = torch.nn.MSELoss().to("cuda")(ray_depth_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_depth_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_depth_masked_coarse_raycolor", loss, mse2psnr(loss)))
        visualizer.accumulate_losses(acc_dict)
        count+=1

    visualizer.print_losses(count)
    psnr = visualizer.get_psnr(opt.test_color_loss_items[0])
    psnr_ray_masked = visualizer.get_psnr(opt.test_color_loss_items[-1])
    visualizer.reset()

    print('--------------------------------Finish Test Rendering--------------------------------')

    #report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr"] if lpips else ["psnr"], [i for i in range(0, total_num, opt.test_num_step)], \
    report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr"] if lpips else ["psnr"], seq_ids, vids, \
                   imgStr="step-%04d-%04d-{}.png".format(opt.visual_items[0]), gtStr="step-%04d-%04d-{}.png".format(opt.visual_items[1]), \
                   writer=writer, iteration=test_steps, epoch=epoch)
    for seq_id in range(len(opt.scans)):
        print (seq_id, opt.scans[seq_id], psnr_list[seq_id])
        print_str = "Scan {}'s psnr is {}, psnr_ray_masked is {}.".format(opt.scans[seq_id], np.mean(np.array(psnr_list[seq_id])), np.mean(np.array(psnr_list_ray_masked[seq_id])))
        cprint.info(print_str)
        visualizer.print_details(print_str)

    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num, opt.test_num_step), test_steps)
        print('--------------------------------Finish generating vid--------------------------------')
    return psnr, psnr_ray_masked

def progressive_distill(model, dataset, visualizer, opt, bg_info, test_steps=0, opacity_thresh=0.7):
    cprint.err('-----------------------------------Progressive Distill-----------------------------------')
    add_xyz_list, add_embedding_list = [],[]
    model.opt.prob = 1

    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    height = dataset.height
    width = dataset.width
    visualizer.reset()

#    if test_steps>=60000:
#        opt.prob_num_step = opt.prob_num_step*2

    #max_num = len(dataset) // opt.prob_num_step
    #frame_ids = list(range(len(dataset)))
    #random.shuffle(frame_ids)
    #frame_ids = frame_ids[:max_num]

    #cprint.err("{}/{} id_lst to add flour".format(len(frame_ids), total_num))
    #cprint.err(frame_ids)
    failed_sample_loc_list = [[] for x in range(len(opt.scans))]

    pro_distill_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if opt.ddp_train else None
    pro_distill_data_loader = torch.utils.data.DataLoader(dataset, \
        sampler=pro_distill_sampler, \
        shuffle=(pro_distill_sampler is None), \
        batch_size=opt.batch_size, \
        num_workers=int(opt.n_threads))
    cprint.info("len of pro_distill_data_loader is {}.".format(len(pro_distill_data_loader)))
    if True:
        for index, data in enumerate(pro_distill_data_loader):
            failed_sample_loc = []
            vid = data["vid"]
            seq_id = data["seq_id"]
            raydir = data['raydir'].clone()
            local_raydir = data['local_raydir'].clone()
            sample_loc_loaded = data['sample_loc_loaded'].clone()
            sample_loc_w_loaded = data['sample_loc_w_loaded'].clone()
            ray_valid_loaded = data['ray_valid_loaded'].clone()
            decoded_features_loaded = data['decoded_features_loaded'].clone()
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

                data['local_raydir'] = local_raydir[:, start:end, :]
                data['ray_valid_loaded'] = ray_valid_loaded[:, start:end, ...]
                data['decoded_features_loaded'] = decoded_features_loaded[:, start:end, ...]
                data['sample_loc_loaded'] = sample_loc_loaded[:, start:end, ...]
                data['sample_loc_w_loaded']=sample_loc_w_loaded[:,start:end,...]

                model.set_input(data)
                output = model.test()
                failed_sample_loc_list[seq_id].append(output["failed_sample_loc"].cuda())
                output["ray_mask"] = output["ray_mask"][..., None]

            local_rank = int(os.environ["LOCAL_RANK"])
#            cprint.info("local_rank {} done {}.".format(local_rank, index))

    for seq_id in range(len(failed_sample_loc_list)):
        failed_sample_loc = torch.cat(failed_sample_loc_list[seq_id])
        add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
        add_dir = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
        add_color = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
        add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)

        cprint.info("failed_sample_loc shape is ....{}".format(failed_sample_loc.shape))

        to_add_pcd_xyz, to_add_pcd_embed, to_add_pcd_color, to_add_pcd_dir = add_flour(failed_sample_loc, candidates=model.neural_points.xyz[seq_id], gap=opt.gap, radius=opt.gap, \
                                     embed = model.neural_points.points_embeding[seq_id], color=model.neural_points.points_color[seq_id], dir=model.neural_points.points_dir[seq_id])

        cprint.info("add level {} res flour xyz {} and embed {} and {} color and {} dir number.".format(opt.gap, to_add_pcd_xyz.shape, to_add_pcd_embed.shape, to_add_pcd_color.shape, to_add_pcd_dir.shape))

        add_xyz       = torch.cat([add_xyz, to_add_pcd_xyz], dim=0)
        add_dir       = torch.cat([add_dir, to_add_pcd_dir], dim=0)
        add_color     = torch.cat([add_color, to_add_pcd_color], dim=0)
        add_embedding = torch.cat([add_embedding, to_add_pcd_embed],dim=0)

        if len(add_xyz) > 0:
            visualizer.save_neural_points("prob_{}_{:04d}".format(seq_id, test_steps), add_xyz, None, None, save_ref=False)
            visualizer.print_details("vis added points to probe folder")
        np.savez('{}/rank{}_seqid{}.npz'.format(opt.resume_dir, local_rank, seq_id), xyz = add_xyz.cpu().numpy(), embed=add_embedding.cpu().numpy(), color=add_color.cpu().numpy(), dir=add_dir.cpu().numpy())
        #add_xyz_list.append(add_xyz)
        #add_embedding_list.append(add_embedding)
    return
    model.opt.prob = 0
    return add_xyz_list, add_embedding_list, None, None, None

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
    cprint.err("{}/{} has holes, id_lst to prune".format(len(frame_ids), total_num), frame_ids, opt.prob_num_step)
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
    opt = TrainOptions().parse()
    basedir = "/mnt/lustre/caiyingjie/logs/checkpoints/"
    init_distributed_mode(opt, False)
    cudnn.benchmark = True
    local_rank = int(os.environ["LOCAL_RANK"])
    writer=None
    #writer = SummaryWriter(os.path.join(basedir, 'summaries', opt.name)) if local_rank==0 else None
    visualizer = Visualizer(opt)

    train_dataset = create_dataset(opt)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if opt.ddp_train else None
    data_loader = torch.utils.data.DataLoader(train_dataset, \
        sampler=sampler, \
        shuffle=(sampler is None), \
        batch_size=opt.batch_size, \
        num_workers=int(opt.n_threads))
    normRw2c = train_dataset.norm_w2c[:3,:3] # torch.eye(3, device="cuda") #
    img_lst, frame_ids = None, None
    best_PSNR, best_PSNR_ray_mask =0.0, 0.0
    best_iter, best_iter_ray_mask =0,0
    points_xyz_all, points_xyz_all_list=None, None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        if len([n for n in glob.glob(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth") if os.path.isfile(n)]) > 0:
            resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if opt.resume_iter == "best":
                opt.resume_iter = "latest"
            resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
            if resume_iter is None:
                epoch_count = 1
                total_steps = 0
                visualizer.print_details("No previous checkpoints, start from scratch!!!!")
            else:
                opt.resume_iter = resume_iter
                cprint.warn("init {} load.".format(os.path.join(resume_dir, '{}_states.pth'.format(resume_iter))))
                states = torch.load(
                    os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location='cpu')
                epoch_count = states['epoch_count']
                total_steps = states['total_steps']
                opt.gap = states['gap']
                if "frame_ids" in states:
                    frame_ids = states['frame_ids']
                else:
                    cprint.err('| no frame_ids in model |')
                best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
                best_iter = states['best_iter'] if 'best_iter' in states else best_iter
                best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                visualizer.print_details('Continue training from {} epoch'.format(opt.resume_iter))
                visualizer.print_details(f"Iter: {total_steps}")
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                del states
            opt.mode = 2
            opt.load_points=1
            opt.resume_dir=resume_dir
            opt.resume_iter = resume_iter
            opt.is_train=True
            model = create_model(opt)

            state_dict = torch.load(os.path.join(resume_dir, '{}_net_ray_marching.pth'.format(resume_iter)), map_location='cpu')
            points_xyz_all_list, points_embedding_all_list, points_color_all_list, points_dir_all_list = [],[],[],[]
            add_xyz_list, add_embedding_list, add_color_list, add_dir_list = [],[],[],[]
            for scan_idx, scan in enumerate(opt.scans):
                points_xyz_all_list.append(state_dict["module.neural_points.xyz."+str(scan_idx)].cuda())
                points_embedding_all_list.append(state_dict["module.neural_points.points_embeding."+str(scan_idx)].cuda())
                points_color_all_list.append(state_dict["module.neural_points.points_color."+str(scan_idx)].cuda())
                points_dir_all_list.append(state_dict["module.neural_points.points_dir."+str(scan_idx)].cuda())

                add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
                add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)
                add_color = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
                add_dir = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
                files = glob.glob(os.path.join(opt.resume_dir, "*seqid"+str(scan_idx)+'.npz'))
                for file_name in files:
                    data = np.load(file_name)
                    add_xyz       = torch.cat([add_xyz, torch.as_tensor(data["xyz"]).cuda()], dim=0)
                    add_embedding = torch.cat([add_embedding, torch.as_tensor(data["embed"]).cuda()],dim=0)
                    add_color = torch.cat([add_color, torch.as_tensor(data["color"]).cuda()],dim=0)
                    add_dir = torch.cat([add_dir, torch.as_tensor(data["dir"]).cuda()],dim=0)
                add_xyz_list.append(add_xyz)
                add_embedding_list.append(add_embedding)
                add_color_list.append(add_color)
                add_dir_list.append(add_dir)

            model.set_points(points_xyz_all_list, points_embedding_all_list, points_color=points_color_all_list, points_dir=points_dir_all_list, Rw2c=normRw2c.cuda(), setup_optimizer=False)
            model.grow_points(add_xyz_list, add_embedding_list, add_color_list, add_dir_list, None, dstdir=resume_dir , epoch=epoch_count)
            model.init_scheduler(total_steps, opt)
            del points_xyz_all_list, points_embedding_all_list, add_xyz_list, add_embedding_list, points_color_all_list, points_dir_all_list, add_color_list, add_dir_list
        else:
            load_points = opt.load_points
            opt.is_train = False
            opt.mode = 1
            opt.load_points = 0
            model = create_model(opt)
            model.setup(opt)
            model.eval()
            if load_points == 2:
                points_xyz_all_list, points_embedding_all_list, points_color_all_list, points_dir_all_list = [],[],[],[]
                for scan_idx, scan in enumerate(opt.scans):
                    points_xyz_all = train_dataset.get_candicates(scan)
                    cprint.info('origin pcd shape {}.'.format(points_xyz_all.shape))
                    if opt.ranges[0] > -99.0:
                        ranges = torch.as_tensor(opt.ranges, device=points_xyz_all.device, dtype=torch.float32)
                        mask = torch.prod(
                            torch.logical_and(points_xyz_all[..., :3] >= ranges[None, :3], points_xyz_all[..., :3] <= ranges[None, 3:]),
                            dim=-1) > 0
                        points_xyz_all = points_xyz_all[mask]
                    if opt.embed_init_type=='random':
                        points_embedding_all = torch.randn([1, len(points_xyz_all), opt.point_features_dim], device=points_xyz_all.device, dtype=torch.float32)
                    else:
                        campos, camdir = train_dataset.get_campos_ray(scan, scan_idx)
                        base_num = torch.sum(torch.tensor(train_dataset.seq_ids) < scan_idx)
                        cam_ind = nearest_view(campos, camdir, points_xyz_all)+base_num
                        unique_cam_ind = torch.unique(cam_ind)
                        print(scan_idx,"unique_cam_ind", unique_cam_ind.shape, unique_cam_ind)
                        points_xyz_all = [points_xyz_all[cam_ind[:,0]==unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]
                        featuredim = opt.point_features_dim
                        points_embedding_all = torch.zeros([1, 0, featuredim], device=unique_cam_ind.device, dtype=torch.float32)
                        points_color_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
                        points_dir_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
                        points_conf_all = torch.zeros([1, 0, 1], device=unique_cam_ind.device, dtype=torch.float32)
                        print("extract points embeding & colors", )
                        for i in tqdm(range(len(unique_cam_ind))):
                            id = unique_cam_ind[i]
                            batch = train_dataset.get_item(id, full_img=True, npz=False)
                            HDWD = [train_dataset.height, train_dataset.width]
                            c2w = batch["c2w"][0].cuda()
                            w2c = torch.inverse(c2w)
                            intrinsic = batch["intrinsic"].cuda()
                            # cam_xyz_all 252, 4
                            cam_xyz_all = (torch.cat([points_xyz_all[i], torch.ones_like(points_xyz_all[i][...,-1:])], dim=-1) @ w2c.transpose(0,1))[..., :3]
                            embedding, color, dir, conf = model.query_embedding(HDWD, cam_xyz_all[None,...], None, batch['images'].cuda(), c2w[None, None,...], w2c[None, None,...], intrinsic[:, None,...], 0, pointdir_w=True)
                            conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                            points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                            points_color_all = torch.cat([points_color_all, color], dim=1)
                            points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                            points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                            # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
                        points_xyz_all=torch.cat(points_xyz_all, dim=0)
                        #visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
                    #cprint.info("initial pcd info shape {} and {}".format(points_embedding_all.shape, points_xyz_all.shape))

                    points_embedding_all_list.append(points_embedding_all)
                    points_xyz_all_list.append(points_xyz_all)
                    points_color_all_list.append(points_color_all)
                    points_dir_all_list.append(points_dir_all)

            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train = True
            opt.mode = 2
            model = create_model(opt)
            model.set_points(points_xyz_all_list, points_embedding_all_list, points_color=points_color_all_list, points_dir=points_dir_all_list, Rw2c=normRw2c.cuda() if opt.load_points < 1 and opt.normview != 3 else None)
            epoch_count = 1
            total_steps = 0
            del points_xyz_all, points_embedding_all, points_xyz_all_list, points_embedding_all_list, points_color_all_list, points_dir_all_list

    if opt.ddp_train:
        model.set_ddp()
    model.setup(opt, train_len=len(train_dataset))
    model.train()
    opt.resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"
    test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if opt.ddp_train else None
    test_data_loader = torch.utils.data.DataLoader(test_dataset, \
        sampler=test_sampler, \
        shuffle=(sampler is None), \
        batch_size=opt.batch_size, \
        num_workers=int(opt.n_threads))
    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    if total_steps > 0:
        for scheduler in model.schedulers:
            for i in range(total_steps):
                scheduler.step()
    fg_masks = None
    bg_ray_train_lst, bg_ray_test_lst = [], []
    test_bg_info, render_bg_info = None, None
    img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = None, None, None, None, None

    real_start=total_steps
    train_random_sample_size = opt.random_sample_size
    #for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
    if False==True:
        torch.cuda.empty_cache()
        #test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
        model.opt.is_train = 0
        model.opt.no_loss = 1
        with torch.no_grad():
            test_psnr, test_psnr_ray_mask = test(model, test_data_loader, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=False, writer=writer, epoch=0, height=test_dataset.height, width=test_dataset.width)
        model.opt.no_loss = 0
        model.opt.is_train = 1
        #del test_dataset
        best_iter = total_steps if test_psnr > best_PSNR else best_iter
        best_PSNR = max(test_psnr, best_PSNR)
        best_iter_ray_mask = total_steps if test_psnr_ray_mask > best_PSNR_ray_mask else best_iter_ray_mask
        best_PSNR_ray_mask = max(test_psnr_ray_mask, best_PSNR_ray_mask)
        visualizer.print_details(f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
        visualizer.print_details(f"test at iter {total_steps}, PSNR_ray_mask: {test_psnr_ray_mask}, best_PSNR_ray_mask: {best_PSNR_ray_mask}, best_iter: {best_iter_ray_mask}")
        exit()
    if local_rank==0:
        cprint.warn("| current opt.gap is {}.".format(opt.gap))
    for epoch in range(epoch_count+1, opt.maximum_epoch+1):
        #epoch_start_time = time.time()
        if opt.ddp_train:
            sampler.set_epoch(epoch)
        for i, data in enumerate(data_loader):
            total_steps += 1
            model.set_input(data)
            model.optimize_parameters(total_steps=total_steps)
            losses = model.get_current_losses()
            visualizer.accumulate_losses(losses)

            if opt.lr_policy.startswith("iter"):
                model.update_learning_rate(opt=opt, total_steps=total_steps)

            if total_steps and total_steps % opt.print_freq == 0:
                if opt.show_tensorboard:
                    visualizer.plot_current_losses_with_tb(total_steps, losses)
                if local_rank==0:
                    visualizer.print_losses(total_steps, writer, epoch)
                visualizer.reset()

            model.train()
        #if local_rank==0:
        #    cprint.warn("epoch training time is {}.".format(time.time()-epoch_start_time))

        torch.cuda.empty_cache()
        if local_rank==0 and epoch % opt.save_iter_freq == 0 and total_steps > 0 and epoch!=epoch_count and (epoch not in opt.prob_tiers):
            other_states = {
                "best_PSNR": best_PSNR,
                "best_iter": best_iter,
                'epoch_count': epoch,
                'total_steps': total_steps,
                'gap': opt.gap,
            }
            visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
            model.save_networks(total_steps, epoch, other_states)
        if  epoch!=epoch_count and epoch % opt.test_freq == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0:
            torch.cuda.empty_cache()
            model.opt.is_train = 0
            model.opt.no_loss = 1
            with torch.no_grad():
                test_psnr, test_psnr_ray_mask = test(model, test_data_loader, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=False, writer=writer, epoch=epoch, height=test_dataset.height, width=test_dataset.width)
            model.opt.no_loss = 0
            model.opt.is_train = 1
            best_iter = total_steps if test_psnr > best_PSNR else best_iter
            best_PSNR = max(test_psnr, best_PSNR)
            best_iter_ray_mask = total_steps if test_psnr_ray_mask > best_PSNR_ray_mask else best_iter_ray_mask
            best_PSNR_ray_mask = max(test_psnr_ray_mask, best_PSNR_ray_mask)
            visualizer.print_details(f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
            visualizer.print_details(f"test at iter {total_steps}, PSNR_ray_mask: {test_psnr_ray_mask}, best_PSNR_ray_mask: {best_PSNR_ray_mask}, best_iter: {best_iter_ray_mask}")

        if epoch!=epoch_count and opt.prob_freq > 0 and real_start != total_steps and (epoch in opt.prob_tiers) and total_steps < (opt.prob_maximum_step - 1) and total_steps > 0 and opt.progressive_distill:
            tier = np.sum(np.asarray(opt.prob_tiers) < total_steps)
            model.opt.is_train = 0
            model.opt.no_loss = 1
            with torch.no_grad():
                prob_opt = copy.deepcopy(test_opt)
                opt.gap = opt.gap/3

                if local_rank==0:
                    max_num = len(train_dataset) // opt.prob_num_step + 1
                    next_frame_ids = list(range(len(train_dataset)))
                    random.shuffle(next_frame_ids)
                    next_frame_ids = next_frame_ids[:max_num]
                    cprint.warn("{}/{} id_lst to add flour at NEXT step with gap {}.".format(len(next_frame_ids), len(train_dataset), opt.gap/3))
                    cprint.warn(next_frame_ids)
                    other_states = {
                        "best_PSNR": best_PSNR,
                        "best_iter": best_iter,
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                        'gap': opt.gap,
                        'frame_ids': next_frame_ids,
                    }
                    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    model.save_networks(total_steps, epoch, other_states)

                prob_opt.name = opt.name
                prob_opt.gap = opt.gap
                train_dataset.opt.random_sample = "no_crop"
                if opt.prob_mode <= 0 and 1==2:
                    train_dataset.opt.random_sample_size = min(32, train_random_sample_size)
                    prob_dataset = train_dataset
                else:
                    if frame_ids is None:
                        max_num = len(train_dataset) // opt.prob_num_step + 1
                        frame_ids = list(range(len(train_dataset)))
                        frame_ids = frame_ids[::opt.prob_num_step]
                        frame_ids = frame_ids[:max_num]

                    cprint.info(len(frame_ids), frame_ids)
                    prob_dataset = create_diy_dataset(frame_ids, test_opt, opt, total_steps, test_num_step=1)

                model.eval()
                cprint.info('add flour at resolution {}'.format(prob_opt.gap))
                progressive_distill(model, prob_dataset, Visualizer(prob_opt), prob_opt, None, test_steps=total_steps, opacity_thresh=opt.prob_thresh)
                exit()
    if local_rank==0:
        writer.close()
    del train_dataset
    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
    model.save_networks(total_steps, epoch, other_states)

    torch.cuda.empty_cache()
    test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=1)
    model.opt.no_loss = 1
    model.opt.is_train = 0

    visualizer.print_details("full datasets test:")
    with torch.no_grad():
        test_psnr = test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, gen_vid=True, lpips=True)
    best_iter = total_steps if test_psnr > best_PSNR else best_iter
    best_PSNR = max(test_psnr, best_PSNR)
    visualizer.print_details(
        f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
    exit()

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

def create_diy_dataset(id_list, test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["diy"]
    test_opt.split = "diy"
    test_opt.name = opt.name + "/diy_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt, id_list)
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
    main()
