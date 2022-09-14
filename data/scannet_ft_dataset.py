from models.mvs.mvs_utils import read_pfm
from pytorch3d.ops import ball_query
import os
import numpy as np
import cv2
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import h5py
import models.mvs.mvs_utils as mvs_utils
from data.base_dataset import BaseDataset
import configparser
from cprint import *
from os.path import join
from .data_utils import get_dtu_raydir
from plyfile import PlyData, PlyElement
import io
import random

FLIP_Z = np.asarray([
    [1,0,0],
    [0,1,0],
    [0,0,-1],
], dtype=np.float32)

def colorjitter(img, factor):
    img = F.adjust_saturation(img, factor[2])
    img = F.adjust_hue(img, factor[3]-1.0)
    return img
def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    c2w = torch.FloatTensor(c2w)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions

class ScannetFtDataset(BaseDataset):
    def initialize(self, opt, frame_ids=None, img_wh=[1296, 968], downSample=1.0, max_len=-1, norm_w2c=None, norm_c2w=None):
        self.opt = opt
        self.data_dir = opt.data_root
        self.scans = opt.scans
        self.split = opt.split

        self.img_wh = (int(opt.img_wh[0] * downSample), int(opt.img_wh[1] * downSample))
        self.downSample = downSample

        self.scale_factor = 1.0 / 1.0
        self.max_len = max_len
        self.near_far = [opt.near_plane, opt.far_plane]
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        if not self.opt.bg_color or self.opt.bg_color == 'black':
            self.bg_color = (0, 0, 0)
        elif self.opt.bg_color == 'white':
            self.bg_color = (1, 1, 1)
        elif self.opt.bg_color == 'red':
            self.bg_color = (1, 0, 0)
        elif self.opt.bg_color == 'random':
            self.bg_color = 'random'
        else:
            self.bg_color = [float(one) for one in self.opt.bg_color.split(",")]

        self.define_transforms()

        self.id_list, self.id_list_name, self.image_paths, self.seq_ids = [],[],[],[]
        self.intrinsics, self.depth_intrinsics = [],[]
        for seq_id, scan in enumerate(self.scans):
            seq_len = self.build_init_metas(scan)
            self.seq_ids.extend([seq_id]*seq_len)

            img = Image.open(os.path.join(self.data_dir, scan, "exported/color/0.jpg"))
            ori_img_shape = list(self.transform(img).shape)

            f_url_intrinsic = os.path.join(self.data_dir, scan, "exported/intrinsic/intrinsic_color.txt")
            f_url_depth_intrinsic = os.path.join(self.data_dir, scan, "exported/intrinsic/intrinsic_depth.txt")
            intrinsic = np.loadtxt(f_url_intrinsic).astype(np.float32)[:3,:3]
            depth_intrinsic = np.loadtxt(f_url_depth_intrinsic).astype(np.float32)[:3, :3]
            intrinsic[0, :] *= (self.width / ori_img_shape[2])
            intrinsic[1, :] *= (self.height / ori_img_shape[1])

            self.intrinsics.append(intrinsic)
            self.depth_intrinsics.append(depth_intrinsic)

        if frame_ids is not None:
            self.id_list = [self.id_list[i] for i in frame_ids]
            self.id_list_name = [self.id_list_name[i] for i in frame_ids]
            self.seq_ids = [self.seq_ids[i] for i in frame_ids]

        mapIndexPosition = list(zip(self.id_list, self.id_list_name, self.seq_ids))
        random.shuffle(mapIndexPosition)
        self.id_list, self.id_list_name, self.seq_ids = zip(*mapIndexPosition)

        self.view_id_list=[]
        self.norm_w2c, self.norm_c2w = torch.eye(4, device="cuda", dtype=torch.float32), torch.eye(4, device="cuda", dtype=torch.float32)
        self.total = len(self.id_list)
        self.ray_valid_loaded = []
        self.sample_loc_loaded = []
        self.sample_loc_w_loaded=[]
        self.decoded_features_loaded=[]

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # ['random', 'random2', 'patch'], default: no random sample
        parser.add_argument('--random_sample',
                            type=str,
                            default='none',
                            help='random sample pixels')
        parser.add_argument('--random_sample_size',
                            type=int,
                            default=1024,
                            help='number of random samples')
        parser.add_argument('--init_view_num',
                            type=int,
                            default=3,
                            help='number of random samples')
        parser.add_argument('--edge_filter',
                            type=int,
                            default=3,
                            help='number of random samples')
        parser.add_argument('--shape_id', type=int, default=0, help='shape id')
        parser.add_argument('--trgt_id', type=int, default=0, help='shape id')
        parser.add_argument('--num_nn',
                            type=int,
                            default=1,
                            help='number of nearest views in a batch')
        parser.add_argument(
            '--near_plane',
            type=float,
            default=0.5,
            help=
            'Near clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument(
            '--far_plane',
            type=float,
            default=5.0,
            help=
            'Far clipping plane, by default it is computed according to the distance of the camera '
        )

        parser.add_argument(
            '--bg_color',
            type=str,
            default="white",
            help=
            'background color, white|black(None)|random|rgb (float, float, float)'
        )

        parser.add_argument(
            '--scan',
            type=str,
            default="scan1",
            help=''
        )

        parser.add_argument('--inverse_gamma_image',
                            type=int,
                            default=-1,
                            help='de-gamma correct the input image')
        parser.add_argument('--pin_data_in_memory',
                            type=int,
                            default=-1,
                            help='load whole data in memory')
        parser.add_argument('--normview',
                            type=int,
                            default=0,
                            help='load whole data in memory')
        parser.add_argument(
            '--id_range',
            type=int,
            nargs=3,
            default=(0, 385, 1),
            help=
            'the range of data ids selected in the original dataset. The default is range(0, 385). If the ids cannot be generated by range, use --id_list to specify any ids.'
        )
        parser.add_argument(
            '--id_list',
            type=int,
            nargs='+',
            default=None,
            help=
            'the list of data ids selected in the original dataset. The default is range(0, 385).'
        )
        parser.add_argument(
            '--split',
            type=str,
            default="train",
            help=
            'train, val, test'
        )
        parser.add_argument("--half_res", action='store_true',
                            help='load blender synthetic data at 400x400 instead of 800x800')
        parser.add_argument("--testskip", type=int, default=8,
                            help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
        parser.add_argument('--dir_norm',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--train_load_num',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')
        parser.add_argument(
            '--img_wh',
            type=int,
            nargs=2,
            default=(640, 480),
            help='resize target of the image'
        )
        return parser

    def normalize_cam(self, w2cs, c2ws):
        index = 0
        return w2cs[index], c2ws[index]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def build_init_metas(self, scan=None):
        colordir = os.path.join(self.data_dir, scan, "exported/color")
        image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]

        image_paths = [os.path.join(self.data_dir, scan, "exported/color/{}.jpg".format(i)) for i in range(len(image_paths))]
        #self.image_paths.extend(image_paths)
        all_id_list = self.filter_valid_id(list(range(len(image_paths))), scan)
        self.all_id_list=all_id_list #### hahahahaha

        step=5
        train_id_list = all_id_list[::step]
        test_id_list = [all_id_list[i] for i in range(len(all_id_list)) if (i % step) !=0] if self.opt.test_num_step != 0 else all_id_list
        if len(self.opt.scans)>1:
            test_id_list = test_id_list[::5]
        else:
            test_id_list = test_id_list[::50]

        if self.opt.all_frames:
            id_list = train_id_list+test_id_list
        else:
            id_list = train_id_list if self.split=="train" or self.split=="diy" else test_id_list
        cprint.warn("scane {}'s data info.".format(scan))
        print("all_id_list",len(all_id_list))
        print("test_id_list",len(test_id_list))
        print("train_id_list",len(train_id_list))
        self.id_list.extend(id_list)
        self.id_list_name.extend([scan]*len(id_list))
        return len(id_list)

    def filter_valid_id(self, id_list, scan):
        empty_lst=[]
        for id in id_list:
            f_url = os.path.join(self.data_dir, scan, "exported/pose", "{}.txt".format(id))
            if self.opt.load_type == 'ceph':
                body = client.get(f_url, update_cache=True)
                f_url = io.BytesIO(body)
            c2w = np.loadtxt(f_url).astype(np.float32)
            if np.max(np.abs(c2w)) < 30:
                empty_lst.append(id)
        return empty_lst

    def get_campos_ray(self, scan=None, scan_idx=None):
        centerpixel=np.asarray(self.img_wh).astype(np.float32)[None,:] // 2
        camposes=[]
        centerdirs=[]
        for id, scan_name in zip(self.id_list, self.id_list_name):
            if scan_name!=scan:
                continue
            f_url = os.path.join(self.data_dir, scan, "exported/pose", "{}.txt".format(id))
            if self.opt.load_type == 'ceph':
                body = client.get(f_url, update_cache=True)
                f_url = io.BytesIO(body)
            c2w = np.loadtxt(f_url).astype(np.float32)
            campos = c2w[:3, 3]
            camrot = c2w[:3,:3]
            _, raydir = get_dtu_raydir(centerpixel, self.intrinsics[scan_idx], camrot, True)
            camposes.append(campos)
            centerdirs.append(raydir)
        camposes=np.stack(camposes, axis=0) # 2091, 3
        centerdirs=np.concatenate(centerdirs, axis=0) # 2091, 3
        return torch.as_tensor(camposes, device="cuda", dtype=torch.float32), torch.as_tensor(centerdirs, device="cuda", dtype=torch.float32)


    def build_proj_mats(self, list=None, norm_w2c=None, norm_c2w=None):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        list = self.id_list if list is None else list

        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        self.focal = focal
        self.near_far = np.array([2.0, 6.0])
        for vid in list:
            frame = self.meta['frames'][vid]
            c2w = np.array(frame['transform_matrix']) # @ self.blender2opencv
            if norm_w2c is not None:
                c2w = norm_w2c @ c2w
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)

            intrinsic = np.array([[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mats += [(proj_mat_l, self.near_far)]

        proj_mats, intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds


    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_depth(self, filepath):
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000
        depth_im[depth_im > 8.0] = 0
        depth_im[depth_im < 0.3] = 0
        return depth_im


    def load_init_depth_points(self, device="cuda", vox_res=0, scan=None, seq_id=None):
        file_path = os.path.join(self.data_dir, scan, scan+'_voxelized_pcd.npy')
        if os.path.exists(file_path):
            print(file_path, '------- exist------')
            world_xyz_all = torch.from_numpy(np.load(file_path)).to(device)
            return world_xyz_all

        py, px = torch.meshgrid(
            torch.arange(0, 480, dtype=torch.float32, device=device),
            torch.arange(0, 640, dtype=torch.float32, device=device))
        img_xy = torch.stack([px, py], dim=-1) # [480, 640, 2]
        reverse_intrin = torch.inverse(torch.as_tensor(self.depth_intrinsics[seq_id])).t().to(device)
        world_xyz_all = torch.zeros([0,3], device=device, dtype=torch.float32)
        for i in tqdm(range(len(self.all_id_list))):
            id = self.all_id_list[i]
            c2w = torch.as_tensor(np.loadtxt(os.path.join(self.data_dir, scan, "exported/pose", "{}.txt".format(id))).astype(np.float32), device=device, dtype=torch.float32)  #@ self.blender2opencv
            # 480, 640, 1
            depth = torch.as_tensor(self.read_depth(os.path.join(self.data_dir, scan, "exported/depth/{}.png".format(id))), device=device)[..., None]
            cam_xy =  img_xy * depth
            cam_xyz = torch.cat([cam_xy, depth], dim=-1)
            cam_xyz = cam_xyz @ reverse_intrin
            cam_xyz = cam_xyz[cam_xyz[...,2] > 0,:]
            cam_xyz = torch.cat([cam_xyz, torch.ones_like(cam_xyz[...,:1])], dim=-1)
            world_xyz = (cam_xyz.view(-1,4) @ c2w.t())[...,:3]
            if vox_res > 0:
                world_xyz = mvs_utils.construct_vox_points_xyz(world_xyz, vox_res)
                # print("world_xyz", world_xyz.shape)
            world_xyz_all = torch.cat([world_xyz_all, world_xyz], dim=0)
        if self.opt.ranges[0] > -99.0:
            ranges = torch.as_tensor(self.opt.ranges, device=world_xyz_all.device, dtype=torch.float32)
            mask = torch.prod(torch.logical_and(world_xyz_all >= ranges[None, :3], world_xyz_all <= ranges[None, 3:]), dim=-1) > 0
            world_xyz_all = world_xyz_all[mask]
        return world_xyz_all


    def __len__(self):
        if self.split == 'train':
            return len(self.id_list) if self.max_len <= 0 else self.max_len
        return len(self.id_list) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'NerfSynthFtDataset'


    def __del__(self):
        print("end loading")

    def normalize_rgb(self, data):
        # to unnormalize image for visualization
        # data C, H, W
        C, H, W = data.shape
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (data - mean) / std


    def get_init_item(self, idx, crop=False):
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h, alphas = [], [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = self.view_id_dict[i]
            # mvs_images += [self.normalize_rgb(self.blackimgs[vid])]
            # mvs_images += [self.whiteimgs[vid]]
            mvs_images += [self.blackimgs[vid]]
            imgs += [self.whiteimgs[vid]]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            depths_h.append(self.depths[vid])
            alphas.append(self.alphas[vid])
            near_fars.append(near_far)

        for i in range(len(affine_mat)):
            view_proj_mats = []
            ref_proj_inv = affine_mat_inv[i]
            for j in range(len(affine_mat)):
                if i == j:  # reference view
                    view_proj_mats += [np.eye(4)]
                else:
                    view_proj_mats += [affine_mat[j] @ ref_proj_inv]
            # view_proj_mats: 4, 4, 4
            view_proj_mats = np.stack(view_proj_mats)
            proj_mats.append(view_proj_mats[:, :3])
        # (4, 4, 3, 4)
        proj_mats = np.stack(proj_mats)
        imgs = np.stack(imgs).astype(np.float32)
        mvs_images = np.stack(mvs_images).astype(np.float32)

        depths_h = np.stack(depths_h)
        alphas = np.stack(alphas)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = mvs_images  # (V, 3, H, W)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['alphas'] = alphas.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        for key, value in sample.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                    sample[key] = value.unsqueeze(0)

        return sample



    def __getitem__(self, id, crop=False, full_img=False, npz=True):
        item = {}
        vid = self.id_list[id]
        scan = self.id_list_name[id]
        seq_id = self.seq_ids[id]
        item["seq_id"] = seq_id
        image_path = os.path.join(self.data_dir, scan, "exported/color/{}.jpg".format(vid))
        if self.opt.load_type == 'ceph':
            img_bytes = client.get(image_path)
            assert(img_bytes is not None)
            img_mem_view = memoryview(img_bytes)
            img_array = np.frombuffer(img_mem_view, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.resize(img,  (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # (4, h, w)

        f_url = os.path.join(self.data_dir, scan, "exported/pose", "{}.txt".format(vid))
        if self.opt.load_type == 'ceph':
            body = client.get(f_url, update_cache=True)
            f_url = io.BytesIO(body)
        c2w = np.loadtxt(f_url).astype(np.float32)
        intrinsic = self.intrinsics[seq_id]

        width, height = img.shape[2], img.shape[1]
        camrot = (c2w[0:3, 0:3])
        campos = c2w[0:3, 3]

        item["intrinsic"] = intrinsic
        item["campos"] = torch.from_numpy(campos).float()
        item["c2w"] = torch.from_numpy(c2w).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([self.near_far[1]]).view(1, 1)
        item['near'] = torch.FloatTensor([self.near_far[0]]).view(1, 1)
        item['h'] = height
        item['w'] = width
        item['id'] = id
        item['vid'] = vid
        # bounding box
        margin = self.opt.edge_filter
        if full_img:
            item['images'] = img[None,...].clone()
        gt_image = np.transpose(img, (1, 2, 0))
        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(margin, width - margin - subsamplesize + 1)
            indy = np.random.randint(margin, height - margin - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(margin,
                                   width-margin,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(margin,
                                   height-margin,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(margin,
                                   width - margin - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(margin,
                                   height - margin - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(margin, width - margin).astype(np.float32),
                np.arange(margin, height- margin).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        item["pixel_idx"] = pixelcoords
        # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
        local_raydir, raydir = get_dtu_raydir(pixelcoords, item["intrinsic"], camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        item['local_raydir'] = torch.from_numpy(local_raydir).float().reshape(-1,3)
        gt_image = gt_image[py.astype(np.int32), px.astype(np.int32)]
        gt_image = np.reshape(gt_image, (-1, 3))
        item['gt_image'] = gt_image

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)
        if self.opt.progressive_distill and npz and self.opt.pseudo_gt_load_type!='online':
            data = np.load(os.path.join(self.opt.data_root, scan, 'pseudo_gt', 'results_pointnerf_'+str(scan)+'_'+str(vid)+'.npz'))
            item["sample_loc_loaded"] = data['sample_loc'][py.astype(np.int32), px.astype(np.int32),...].reshape(-1, self.opt.SR, 3)
            item["sample_loc_w_loaded"]=data["sample_loc_w"][py.astype(np.int32), px.astype(np.int32),...].reshape(-1, self.opt.SR, 3)
            item["ray_valid_loaded"] = data["ray_valid"][py.astype(np.int32), px.astype(np.int32),...].reshape(-1, self.opt.SR)
            item["decoded_features_loaded"] = data["decoded_features"][py.astype(np.int32), px.astype(np.int32),...].reshape(-1, self.opt.SR, 4)
        return item

    def get_candicates(self, scan=None):
        file_path = os.path.join(self.opt.data_root, scan, 'init_candidates.npy')
        cprint.warn(file_path)
        if self.opt.load_type=='ceph':
            body = client.get(file_path, update_cache=True)
            if not body:
                LOG.warn('can not get content from %s', file_path)
            file_path = io.BytesIO(body)
            filtered_candidates = torch.from_numpy(np.load(file_path)).cuda()
            return filtered_candidates
        elif os.path.exists(file_path):
            filtered_candidates = torch.from_numpy(np.load(file_path)).cuda()
            return filtered_candidates
        center = torch.tensor([3.7269, 3.4063, 1.2413]).cuda()
        whl = torch.tensor([8.2886, 8.1767, 3.0916]).cuda()

        range_min,range_max = center-whl/2, center+whl/2

        xs = torch.arange(range_min[0], range_max[0], self.opt.gap, device='cuda')
        ys = torch.arange(range_min[1], range_max[1], self.opt.gap, device='cuda')
        zs = torch.arange(range_min[2], range_max[2], self.opt.gap, device='cuda')

        candidates = torch.cartesian_prod(xs, ys, zs)

        idx_flag = torch.zeros(len(candidates)).cuda()
        for vid in self.id_list:
            data = np.load(os.path.join(self.opt.data_root, scan, 'pseudo_gt', 'results_pointnerf_'+str(scan)+'_'+str(vid)+'.npz'))
            sample_loc_w_loaded = data["sample_loc_w"]
            ray_valid_loaded = data["ray_valid"]
            sample_loc_w_loaded = torch.tensor(sample_loc_w_loaded[ray_valid_loaded]).cuda()
            idx = ball_query(candidates[None,...], sample_loc_w_loaded[None,...], K=1, radius=0.2).idx.squeeze() #N*p1*1
            idx_flag[idx>0]=1
        filtered_candidates = candidates[idx_flag>0]
        np.save(file_path, filtered_candidates.cpu().numpy())
        cprint.warn('scene {} INIT candidates points.....{}.'.format(scan, filtered_candidates.shape))
        exit()
        return filtered_candidates

    def get_item(self, idx, crop=False, full_img=False, npz=True):
        item = self.__getitem__(idx, crop=crop, full_img=full_img, npz=npz)

        for key, value in item.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                item[key] = value.unsqueeze(0)
        return item

    def get_dummyrot_item(self, idx, crop=False):

        item = {}
        width, height = self.width, self.height

        transform_matrix = self.render_poses[idx]
        camrot = (transform_matrix[0:3, 0:3])
        campos = transform_matrix[0:3, 3]
        focal = self.focal

        item["focal"] = focal
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float()
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        # near far
        if self.opt.near_plane is not None:
            near = self.opt.near_plane
        else:
            near = max(dist - 1.5, 0.02)
        if self.opt.far_plane is not None:
            far = self.opt.far_plane  # near +
        else:
            far = dist + 0.7
        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([far]).view(1, 1)
        item['near'] = torch.FloatTensor([near]).view(1, 1)
        item['h'] = self.height
        item['w'] = self.width


        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(0,
                                   width,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(0,
                                   height,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(0,
                                   width - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(0,
                                   height - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            px, py = self.proportional_select(gt_mask)
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        raydir = get_blender_raydir(pixelcoords, self.height, self.width, focal, camrot, self.opt.dir_norm > 0)
        item["pixel_idx"] = pixelcoords
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)

        for key, value in item.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            item[key] = value.unsqueeze(0)

        return item

