import torch
import torch.nn as nn
from data.load_blender import load_blender_cloud
import numpy as np
from ..helpers.networks import init_seq, positional_encoding
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune_param
from utils.kitti_object import get_lidar_in_image_fov
import os
from cprint import *
class NeuralPoints(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--load_points',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--point_noise',
                            type=str,
                            default="",
                            help='pointgaussian_0.1 | pointuniform_0.1')

        parser.add_argument('--num_point',
                            type=int,
                            default=8192,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--construct_res',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--grid_res',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--cloud_path',
                            type=str,
                            default="",
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--shpnt_jitter',
                            type=str,
                            default="passfunc",
                            help='passfunc | uniform | gaussian')

        parser.add_argument('--point_features_dim',
                            type=int,
                            default=64,
                            help='number of coarse samples')

        parser.add_argument('--gpu_maxthr',
                            type=int,
                            default=1024,
                            help='number of coarse samples')

        parser.add_argument('--z_depth_dim',
                            type=int,
                            default=400,
                            help='number of coarse samples')

        parser.add_argument('--SR',
                            type=int,
                            default=24,
                            help='max shading points number each ray')

        parser.add_argument('--K',
                            type=int,
                            default=32,
                            help='max neural points each group')

        parser.add_argument('--max_o',
                            type=int,
                            default=None,
                            help='max nonempty voxels stored each frustum')

        parser.add_argument('--P',
                            type=int,
                            default=16,
                            help='max neural points stored each block')

        parser.add_argument('--NN',
                            type=int,
                            default=0,
                            help='0: radius search | 1: K-NN after radius search | 2: K-NN world coord after pers radius search')

        parser.add_argument('--radius_limit_scale',
                            type=float,
                            default=5.0,
                            help='max neural points stored each block')

        parser.add_argument('--depth_limit_scale',
                            type=float,
                            default=1.3,
                            help='max neural points stored each block')

        parser.add_argument('--default_conf',
                            type=float,
                            default=-1.0,
                            help='max neural points stored each block')

        parser.add_argument(
            '--vscale',
            type=int,
            nargs='+',
            default=(2, 2, 1),
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--kernel_size',
            type=int,
            nargs='+',
            default=(7, 7, 1),
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--query_size',
            type=int,
            nargs='+',
            default=(0, 0, 0),
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--xyz_grad',
            type=int,
            default=0,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--feat_grad',
            type=int,
            default=1,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--conf_grad',
            type=int,
            default=1,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--color_grad',
            type=int,
            default=1,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--dir_grad',
            type=int,
            default=1,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--feedforward',
            type=int,
            default=0,
            help=
            'vscale is the block size that store several voxels'
        )

        parser.add_argument(
            '--inverse',
            type=int,
            default=0,
            help=
            '1 for 1/n depth sweep'
        )

        parser.add_argument(
            '--point_conf_mode',
            type=str,
            default="0",
            help=
            '0 for only at features, 1 for multi at weight'
        )
        parser.add_argument(
            '--point_color_mode',
            type=str,
            default="0",
            help=
            '0 for only at features, 1 for multi at weight'
        )
        parser.add_argument(
            '--point_dir_mode',
            type=str,
            default="0",
            help=
            '0 for only at features, 1 for multi at weight'
        )
        parser.add_argument(
            '--vsize',
            type=float,
            nargs='+',
            default=(0.005, 0.005, 0.005),
            help=
            'vscale is the block size that store several voxels'
        )
        parser.add_argument(
            '--wcoord_query',
            type=int,
            default="0",
            help=
            '0 for perspective voxels, and 1 for world coord'
        )
        parser.add_argument(
            '--ranges',
            type=float,
            nargs='+',
            default=(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0),
            help='vscale is the block size that store several voxels'
        )

    def __init__(self, num_channels, size, opt, device, checkpoint=None, feature_init_method='rand', reg_weight=0., feedforward=0):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        self.opt = opt
        self.grid_vox_sz = 0
        self.points_conf, self.points_dir, self.points_color, self.eulers, self.Rw2c = None, None, None, None, None
        self.device=device
        self.reg_weight = reg_weight
        self.opt.query_size = self.opt.kernel_size if self.opt.query_size[0] == 0 else self.opt.query_size
        #self.lighting_fast_querier = lighting_fast_querier_w if self.opt.wcoord_query > 0 else lighting_fast_querier_p
        if self.opt.progressive_distill==False or self.opt.all_sample_loc:
            from .query_point_indices_worldcoords import lighting_fast_querier as lighting_fast_querier_w
            self.lighting_fast_querier = lighting_fast_querier_w
            self.querier = self.lighting_fast_querier(device, self.opt)

        if self.opt.load_points == 10:
            cprint.info('checkpoint is {}.'.format(checkpoint))
            if checkpoint:
                saved_features = torch.load(checkpoint, map_location='cpu')
                '''
                #### new
                data = np.load('/mnt/cache/caiyingjie/code/pointnerf_new/run/scene0006_chair_for_seqid1.npz')
                insert_obj = data['xyz']
                extra_embed, extra_color, extra_dir, extra_conf = data['embed'], data['color'], data['dir'], data['dir'][...,0:1]

                insert_obj_x, insert_obj_y, insert_obj_z = insert_obj[...,0], insert_obj[...,1]-3.5, insert_obj[...,2]
                extra_xyz = torch.from_numpy(np.concatenate((insert_obj_x[...,None], insert_obj_y[...,None], insert_obj_z[...,None]), -1))

                scene_name = checkpoint.split('/')[-2]
                self.xyz = nn.Parameter(torch.cat((saved_features["neural_points.xyz"], extra_xyz), 0))
                #np.savetxt(scene_name+'_scene.txt', saved_features["neural_points.xyz"].cpu().detach().numpy())
                np.savetxt(scene_name+'_scene_with_chair_03.50.txt', extra_xyz.cpu().detach().numpy())
                #if scene_name == 'scene0101_00':
                #    exit()
                cprint.info("self.xyz shape {}".format(self.xyz.shape))
                self.points_embeding = nn.Parameter(torch.cat((saved_features["neural_points.points_embeding"], torch.from_numpy(extra_embed[None,...])), 1)) if "neural_points.points_embeding" in saved_features else None
                self.points_conf = nn.Parameter(torch.cat((saved_features["neural_points.points_conf"], torch.from_numpy(extra_conf[None,...])), 1)) if "neural_points.points_conf" in saved_features else None
                self.points_dir = nn.Parameter(torch.cat((saved_features["neural_points.points_dir"], torch.from_numpy(extra_dir[None,...])), 1)) if "neural_points.points_dir" in saved_features else None
                self.points_color = nn.Parameter(torch.cat((saved_features["neural_points.points_color"], torch.from_numpy(extra_color[None,...])), 1)) if "neural_points.points_color" in saved_features else None
                '''
                self.xyz = nn.Parameter(saved_features["neural_points.xyz"])
                cprint.info("self.xyz shape {}".format(self.xyz.shape))
                self.points_embeding = nn.Parameter(saved_features["neural_points.points_embeding"]) if "neural_points.points_embeding" in saved_features else None
                self.points_conf = nn.Parameter(saved_features["neural_points.points_conf"]) if "neural_points.points_conf" in saved_features else None
                self.points_dir = nn.Parameter(saved_features["neural_points.points_dir"]) if "neural_points.points_dir" in saved_features else None
                self.points_color = nn.Parameter(saved_features["neural_points.points_color"]) if "neural_points.points_color" in saved_features else None
                self.eulers = nn.Parameter(saved_features["neural_points.eulers"]) if "neural_points.eulers" in saved_features else None
                self.Rw2c = nn.Parameter(saved_features["neural_points.Rw2c"]) if "neural_points.Rw2c" in saved_features else torch.eye(3, device=self.xyz.device, dtype=self.xyz.dtype)

    def prune(self, thresh):
        mask = self.points_conf[0,...,0] >= thresh
        self.xyz = nn.Parameter(self.xyz[mask, :])
        self.xyz.requires_grad = self.opt.xyz_grad > 0

        if self.points_embeding is not None:
            self.points_embeding = nn.Parameter(self.points_embeding[:, mask, :])
            self.points_embeding.requires_grad = self.opt.feat_grad > 0
        if self.points_conf is not None:
            self.points_conf = nn.Parameter(self.points_conf[:, mask, :])
            self.points_conf.requires_grad = self.opt.conf_grad > 0
        if self.points_dir is not None:
            self.points_dir = nn.Parameter(self.points_dir[:, mask, :])
            self.points_dir.requires_grad = self.opt.dir_grad > 0
        if self.points_color is not None:
            self.points_color = nn.Parameter(self.points_color[:, mask, :])
            self.points_color.requires_grad = self.opt.color_grad > 0
        if self.eulers is not None and self.eulers.dim() > 1:
            self.eulers = nn.Parameter(self.eulers[mask, :])
            self.eulers.requires_grad = False
        if self.Rw2c is not None and self.Rw2c.dim() > 2:
            self.Rw2c = nn.Parameter(self.Rw2c[mask, :])
            self.Rw2c.requires_grad = False
        print("@@@@@@@@@  pruned {}/{}".format(torch.sum(mask==0), mask.shape[0]))

    def grow_points(self, add_xyz, add_embedding, add_color, add_dir, add_conf, add_eulers=None, add_Rw2c=None, dstdir=None, epoch=None):
        local_rank = int(os.environ["LOCAL_RANK"])
        assert type(add_xyz).__name__=='list'
        for seq_id in range(len(add_xyz)):
            #if local_rank==0:
            #    cprint.info("grow_points {} for seq {}.".format(add_xyz[seq_id].shape, seq_id))
            #    np.savetxt(os.path.join(dstdir, "epoch{}_seqid{}_growxyz.txt".format(epoch, seq_id)), add_xyz[seq_id].cpu().numpy())
            self.xyz[seq_id] = nn.Parameter(torch.cat([self.xyz[seq_id], add_xyz[seq_id]], dim=0))

            self.xyz[seq_id].requires_grad = self.opt.xyz_grad > 0
            self.points_embeding[seq_id] = nn.Parameter(torch.cat([self.points_embeding[seq_id], add_embedding[seq_id][None,...]], dim=1))
            self.points_embeding[seq_id].requires_grad = self.opt.feat_grad > 0

            self.points_color[seq_id] = nn.Parameter(torch.cat([self.points_color[seq_id], add_color[seq_id][None,...]], dim=1))
            self.points_color[seq_id].requires_grad = self.opt.feat_grad > 0

            self.points_dir[seq_id] = nn.Parameter(torch.cat([self.points_dir[seq_id], add_dir[seq_id][None,...]], dim=1))
            self.points_dir[seq_id].requires_grad = self.opt.feat_grad > 0

        self.xyz = nn.ParameterList(self.xyz)
        self.points_embeding = nn.ParameterList(self.points_embeding)
        self.points_color = nn.ParameterList(self.points_color)
        self.points_dir = nn.ParameterList(self.points_dir)

    def grow_points_old(self, add_xyz, add_embedding, add_color, add_dir, add_conf, add_eulers=None, add_Rw2c=None):
        # print(self.xyz.shape, self.points_conf.shape, self.points_embeding.shape, self.points_dir.shape, self.points_color.shape)
        self.xyz = nn.Parameter(torch.cat([self.xyz, add_xyz], dim=0))
        self.xyz.requires_grad = self.opt.xyz_grad > 0

        if self.points_embeding is not None:
            self.points_embeding = nn.Parameter(torch.cat([self.points_embeding, add_embedding[None, ...]], dim=1))
            self.points_embeding.requires_grad = self.opt.feat_grad > 0

        if self.points_conf is not None and add_conf is not None:
            self.points_conf = nn.Parameter(torch.cat([self.points_conf, add_conf[None, ...]], dim=1))
            self.points_conf.requires_grad = self.opt.conf_grad > 0

        if self.points_dir is not None and add_dir is not None:
            self.points_dir = nn.Parameter(torch.cat([self.points_dir, add_dir[None, ...]], dim=1))
            self.points_dir.requires_grad = self.opt.dir_grad > 0

        if self.points_color is not None and add_color is not None:
            self.points_color = nn.Parameter(torch.cat([self.points_color, add_color[None, ...]], dim=1))
            self.points_color.requires_grad = self.opt.color_grad > 0

        if self.eulers is not None and self.eulers.dim() > 1:
            self.eulers = nn.Parameter(torch.cat([self.eulers, add_eulers[None,...]], dim=1))
            self.eulers.requires_grad = False

        if self.Rw2c is not None and self.Rw2c.dim() > 2:
            self.Rw2c = nn.Parameter(torch.cat([self.Rw2c, add_Rw2c[None,...]], dim=1))
            self.Rw2c.requires_grad = False

    def set_points(self, points_xyz, points_embeding, points_color=None, points_dir=None, points_conf=None, parameter=False, Rw2c=None, eulers=None):
        #if points_embeding.shape[-1] > self.opt.point_features_dim:
        #    points_embeding = points_embeding[..., :self.opt.point_features_dim]
        #if self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and points_conf is not None:
        #    points_conf = torch.ones_like(points_conf) * self.opt.default_conf
        if parameter:
            assert type(points_xyz).__name__=='list'
            self.points_embeding, self.xyz, self.points_color, self.points_dir, self.points_conf = [],[],[],[],[]
            seq_id = 0
            for points_xyz_i, points_embeding_i, points_color_i, points_dir_i in zip(points_xyz, points_embeding, points_color, points_dir):
                points_xyz_i = nn.Parameter(points_xyz_i)
                points_xyz_i.requires_grad = self.opt.xyz_grad > 0

                points_embeding_i = nn.Parameter(points_embeding_i)
                points_embeding_i.requires_grad = self.opt.feat_grad > 0

                points_color_i = nn.Parameter(points_color_i)
                points_color_i.requires_grad = self.opt.color_grad > 0

                points_dir_i = nn.Parameter(points_dir_i)
                points_dir_i.requires_grad = self.opt.dir_grad > 0

                if points_conf is not None:
                    points_conf_i = nn.Parameter(points_conf[seq_id])
                    points_conf_i.requires_grad = self.opt.dir_grad > 0
                    self.points_conf.append(points_conf_i)

                self.points_embeding.append(points_embeding_i)
                self.xyz.append(points_xyz_i)
                self.points_color.append(points_color_i)
                self.points_dir.append(points_dir_i)
                seq_id += 1

            self.xyz = nn.ParameterList(self.xyz)
            self.points_embeding = nn.ParameterList(self.points_embeding)
            self.points_color = nn.ParameterList(self.points_color)
            self.points_dir = nn.ParameterList(self.points_dir)
            if points_conf is not None:
                self.points_conf = nn.ParameterList(self.points_conf)

        else:
            self.xyz = points_xyz

            if points_conf is not None:
                if "0" in list(self.opt.point_conf_mode):
                    points_embeding = torch.cat([points_conf, points_embeding], dim=-1)
                if "1" in list(self.opt.point_conf_mode):
                    self.points_conf = points_conf

            if points_dir is not None:
                if "0" in list(self.opt.point_dir_mode):
                    points_embeding = torch.cat([points_dir, points_embeding], dim=-1)
                if "1" in list(self.opt.point_dir_mode):
                    self.points_dir = points_dir

            if points_color is not None:
                if "0" in list(self.opt.point_color_mode):
                    points_embeding = torch.cat([points_color, points_embeding], dim=-1)
                if "1" in list(self.opt.point_color_mode):
                    self.points_color = points_color

            self.points_embeding = points_embeding

        if Rw2c is None:
            self.Rw2c = torch.eye(3, device=points_xyz.device, dtype=points_xyz.dtype)
        else:
            self.Rw2c = nn.Parameter(Rw2c)
            self.Rw2c.requires_grad = False


    def editing_set_points(self, points_xyz, points_embeding, points_color=None, points_dir=None, points_conf=None,
                   parameter=False, Rw2c=None, eulers=None):
        if self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and points_conf is not None:
            points_conf = torch.ones_like(points_conf) * self.opt.default_conf

        self.xyz = points_xyz
        self.points_embeding = points_embeding
        self.points_dir = points_dir
        self.points_conf = points_conf
        self.points_color = points_color

        if Rw2c is None:
            self.Rw2c = torch.eye(3, device=points_xyz.device, dtype=points_xyz.dtype)
        else:
            self.Rw2c = Rw2c



    def construct_grid_points(self, xyz):
        # --construct_res' '--grid_res',
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        self.space_edge = torch.max(xyz_max - xyz_min) * 1.1
        xyz_mid = (xyz_max + xyz_min) / 2
        self.space_min = xyz_mid - self.space_edge / 2
        self.space_max = xyz_mid + self.space_edge / 2
        self.construct_vox_sz = self.space_edge / self.opt.construct_res
        self.grid_vox_sz = self.space_edge / self.opt.grid_res

        xyz_shift = xyz - self.space_min[None, ...]
        construct_vox_idx = torch.unique(torch.floor(xyz_shift / self.construct_vox_sz[None, ...]).to(torch.int16), dim=0)
        # print("construct_grid_idx", construct_grid_idx.shape) torch.Size([7529, 3])

        cg_ratio = int(self.opt.grid_res / self.opt.construct_res)
        gx = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gy = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gz = torch.arange(0, cg_ratio+1, device=construct_vox_idx.device, dtype=construct_vox_idx.dtype)
        gx, gy, gz = torch.meshgrid(gx, gy, gz)
        gxyz = torch.stack([gx, gy, gz], dim=-1).view(1, -1, 3)
        sparse_grid_idx = construct_vox_idx[:, None, :] * cg_ratio + gxyz
        # sparse_grid_idx.shape: ([7529, 9*9*9, 3]) -> ([4376896, 3])
        sparse_grid_idx = torch.unique(sparse_grid_idx.view(-1, 3), dim=0).to(torch.int64)
        full_grid_idx = torch.full([self.opt.grid_res+1,self.opt.grid_res+1,self.opt.grid_res+1], -1, device=xyz.device, dtype=torch.int32)
        # full_grid_idx.shape:    ([401, 401, 401])
        full_grid_idx[sparse_grid_idx[...,0], sparse_grid_idx[...,1], sparse_grid_idx[...,2]] = torch.arange(0, sparse_grid_idx.shape[0], device=full_grid_idx.device, dtype=full_grid_idx.dtype)
        xyz = self.space_min[None, ...] + sparse_grid_idx * self.grid_vox_sz
        return xyz, sparse_grid_idx, full_grid_idx


    def null_grad(self):
        self.points_embeding.grad = None
        self.xyz.grad = None


    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.points_embeding, 2))


    def pers2img(self, point_xyz_pers_tensor, pixel_id, pixel_idx_cur, ray_mask, sample_pidx, ranges, h, w, inputs):
        xper = point_xyz_pers_tensor[..., 0].cpu().numpy()
        yper = point_xyz_pers_tensor[..., 1].cpu().numpy()

        x_pixel = np.clip(np.round((xper-ranges[0]) * (w-1) / (ranges[3]-ranges[0])).astype(np.int32), 0, w-1)[0]
        y_pixel = np.clip(np.round((yper-ranges[1]) * (h-1) / (ranges[4]-ranges[1])).astype(np.int32), 0, h-1)[0]

        print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel),
              np.min(y_pixel), sample_pidx.shape,y_pixel.shape)
        background = np.zeros([h, w, 3], dtype=np.float32)
        background[y_pixel, x_pixel, :] = self.points_embeding.cpu().numpy()[0,...]

        background[pixel_idx_cur[0,...,1],pixel_idx_cur[0,...,0],0] = 1.0

        background[y_pixel[sample_pidx[-1]], x_pixel[sample_pidx[-1]], :] = self.points_embeding.cpu().numpy()[0,sample_pidx[-1]]

        gtbackground = np.ones([h, w, 3], dtype=np.float32)
        gtbackground[pixel_idx_cur[0 ,..., 1], pixel_idx_cur[0 , ..., 0],:] = inputs["gt_image"].cpu().numpy()[0][ray_mask[0]>0]

        print("diff sum",np.sum(inputs["gt_image"].cpu().numpy()[0][ray_mask[0]>0]-self.points_embeding.cpu().numpy()[0,sample_pidx[...,1,0][-1]]))

        plt.figure()
        plt.imshow(background)
        plt.figure()
        plt.imshow(gtbackground)
        plt.show()

    def get_point_indices(self, inputs, cam_rot_tensor, cam_pos_tensor, pixel_idx_tensor, near_plane, far_plane, h, w, intrinsic, vox_query=False):

        if self.opt.load_points==10:
            point_xyz_pers_tensor = self.w2pers(self.xyz, cam_rot_tensor, cam_pos_tensor)
        else:
            point_xyz_pers_tensor = self.w2pers(self.xyz_fov, cam_rot_tensor, cam_pos_tensor)
        actual_numpoints_tensor = torch.ones([point_xyz_pers_tensor.shape[0]], device=point_xyz_pers_tensor.device, dtype=torch.int32) * point_xyz_pers_tensor.shape[1]
        ray_dirs_tensor = inputs["raydir"]
        local_ray_dirs_tensor = inputs["local_raydir"]
        if self.opt.load_points==10:
            sample_pidx_tensor, sample_loc_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, ray_mask_tensor, vsize, ranges, raypos_tensor, index_tensor = self.querier.query_points(pixel_idx_tensor, point_xyz_pers_tensor, self.xyz[None,...], actual_numpoints_tensor, h, w, intrinsic, near_plane, far_plane, ray_dirs_tensor, local_ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor)
        else:
            sample_pidx_tensor, sample_loc_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, ray_mask_tensor, vsize, ranges, raypos_tensor, index_tensor = self.querier.query_points(pixel_idx_tensor, point_xyz_pers_tensor, self.xyz_fov[None,...], actual_numpoints_tensor, h, w, intrinsic, near_plane, far_plane, ray_dirs_tensor, local_ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor)

        B, _, SR, K = sample_pidx_tensor.shape
        if vox_query:
            if sample_pidx_tensor.shape[1] > 0:
                sample_pidx_tensor = self.query_vox_grid(sample_loc_w_tensor, self.full_grid_idx, self.space_min, self.grid_vox_sz)
            else:
                sample_pidx_tensor = torch.zeros([B, 0, SR, 8], device=sample_pidx_tensor.device, dtype=sample_pidx_tensor.dtype)

        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor, point_xyz_pers_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, \
            sample_local_ray_dirs_tensor, vsize, raypos_tensor, index_tensor

    def query_vox_grid(self, sample_loc_w_tensor, full_grid_idx, space_min, grid_vox_sz):
        # sample_pidx_tensor = torch.full(sample_loc_w_tensor.shape[:-1]+(8,), -1, device=sample_loc_w_tensor.device, dtype=torch.int64)
        B, R, SR, _ = sample_loc_w_tensor.shape
        vox_ind = torch.floor((sample_loc_w_tensor - space_min[None, None, None, :]) / grid_vox_sz).to(torch.int64) # B, R, SR, 3
        shift = torch.as_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.int64, device=full_grid_idx.device).reshape(1, 1, 1, 8, 3)
        vox_ind = vox_ind[..., None, :] + shift  # B, R, SR, 8, 3
        vox_mask = torch.any(torch.logical_or(vox_ind < 0, vox_ind > self.opt.grid_res).view(B, R, SR, -1), dim=3)
        vox_ind = torch.clamp(vox_ind, min=0, max=self.opt.grid_res).view(-1, 3)
        inds = full_grid_idx[vox_ind[..., 0], vox_ind[..., 1], vox_ind[..., 2]].view(B, R, SR, 8)
        inds[vox_mask, :] = -1
        # -1 for all 8 corners
        inds[torch.any(inds < 0, dim=-1), :] = -1
        return inds.to(torch.int64)

    # def w2pers(self, point_xyz, camrotc2w, campos):
    #     point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
    #     xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
    #     # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
    #     xper = xyz[:, :, 0] / -xyz[:, :, 2]
    #     yper = xyz[:, :, 1] / xyz[:, :, 2]
    #     return torch.stack([xper, yper, -xyz[:, :, 2]], dim=-1)


    def w2pers(self, point_xyz, camrotc2w, campos):
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
        xper = xyz[:, :, 0] / xyz[:, :, 2]
        yper = xyz[:, :, 1] / xyz[:, :, 2]
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)

    def vect2euler(self, xyz):
        yz_norm = torch.norm(xyz[...,1:3], dim=-1)
        e_x = torch.atan2(-xyz[...,1], xyz[...,2])
        e_y = torch.atan2(xyz[...,0], yz_norm)
        e_z = torch.zeros_like(e_y)
        e_xyz = torch.stack([e_x, e_y, e_z], dim=-1)
        return e_xyz

    def euler2Rc2w(self, e_xyz):
        cosxyz = torch.cos(e_xyz)
        sinxyz = torch.sin(e_xyz)
        cxsz = cosxyz[...,0]*sinxyz[...,2]
        czsy = cosxyz[...,2]*sinxyz[...,1]
        sxsz = sinxyz[...,0]*sinxyz[...,2]
        r1 = torch.stack([cosxyz[...,1]*cosxyz[...,2], czsy*sinxyz[...,0] - cxsz, czsy*cosxyz[...,0] + sxsz], dim=-1)
        r2 = torch.stack([cosxyz[...,1]*sinxyz[...,2], cosxyz[...,0]*cosxyz[...,2] + sxsz*sinxyz[...,1], -cosxyz[...,2]*sinxyz[...,0] + cxsz * sinxyz[...,1]], dim=-1)
        r3 = torch.stack([-sinxyz[...,1], cosxyz[...,1]*sinxyz[...,0], cosxyz[...,0]*cosxyz[...,1]], dim=-1)

        Rzyx = torch.stack([r1, r2, r3], dim=-2)
        return Rzyx

    def euler2Rw2c(self, e_xyz):
        c = torch.cos(-e_xyz)
        s = torch.sin(-e_xyz)
        r1 = torch.stack([c[...,1] * c[...,2], -s[...,2], c[...,2]*s[...,1]], dim=-1)
        r2 = torch.stack([s[...,0]*s[...,1] + c[...,0]*c[...,1]*s[...,2], c[...,0]*c[...,2], -c[...,1]*s[...,0]+c[...,0]*s[...,1]*s[...,2]], dim=-1)
        r3 = torch.stack([-c[...,0]*s[...,1]+c[...,1]*s[...,0]*s[...,2], c[...,2]*s[...,0], c[...,0]*c[...,1]+s[...,0]*s[...,1]*s[...,2]], dim=-1)
        Rxyz = torch.stack([r1, r2, r3], dim=-2)
        return Rxyz

    def get_w2c(self, cam_xyz, Rw2c):
        t = -Rw2c @ cam_xyz[..., None] # N, 3
        M = torch.cat([Rw2c, t], dim=-1)
        ones = torch.as_tensor([[[0, 0, 0, 1]]], device=M.device, dtype=M.dtype).expand(len(M),-1, -1)
        return torch.cat([M, ones], dim=-2)

    def get_c2w(self, cam_xyz, Rc2w):
        M = torch.cat([Rc2w, cam_xyz[..., None]], dim=-1)
        ones = torch.as_tensor([[[0, 0, 0, 1]]], device=M.device, dtype=M.dtype).expand(len(M),-1, -1)
        return torch.cat([M, ones], dim=-2)


    # def pers2w(self, point_xyz_pers, camrotc2w, campos):
    #     #     point_xyz_pers    B X M X 3
    #
    #     x_pers = point_xyz_pers[..., 0] * point_xyz_pers[..., 2]
    #     y_pers = - point_xyz_pers[..., 1] * point_xyz_pers[..., 2]
    #     z_pers = - point_xyz_pers[..., 2]
    #     xyz_c = torch.stack([x_pers, y_pers, z_pers], dim=-1)
    #     xyz_w_shift = torch.sum(xyz_c[...,None,:] * camrotc2w, dim=-1)
    #     # print("point_xyz_pers[..., 0, 0]", point_xyz_pers[..., 0, 0].shape, point_xyz_pers[..., 0, 0])
    #     ray_dirs = xyz_w_shift / (torch.linalg.norm(xyz_w_shift, dim=-1, keepdims=True) + 1e-7)
    #
    #     xyz_w = xyz_w_shift + campos[:, None, :]
    #     return xyz_w, ray_dirs

    def passfunc(self, input, vsize):
        return input

    def pointgaussian(self, input, std):
        M, C = input.shape
        input = torch.normal(mean=input, std=std)
        return input

    def pointuniform(self, input, std):
        M, C = input.shape
        jitters = torch.rand([M, C], dtype=torch.float32, device=input.device) - 0.5
        input = input + jitters * std * 2
        return input

    def pointuniformadd(self, input, std):
        addinput = self.pointuniform(input, std)
        return torch.cat([input,addinput], dim=0)

    def pointuniformdouble(self, input, std):
        input = self.pointuniform(torch.cat([input,input], dim=0), std)
        return input

    def forward(self, inputs):
        #mask = np.load('/mnt/cache/caiyingjie/code/pointnerf_new/run/610.npy')[1]
        pixel_idx, camrotc2w, campos, near_plane, far_plane, h, w, intrinsic = inputs["pixel_idx"].to(torch.int32), inputs["camrotc2w"], inputs["campos"], inputs["near"], inputs["far"], inputs["h"], inputs["w"], inputs["intrinsic"]
        if self.opt.load_points==10:
            sample_pidx, sample_loc, ray_mask_tensor, point_xyz_pers_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, vsize, raypos_tensor, index_tensor = self.get_point_indices(inputs, camrotc2w, campos, pixel_idx, torch.min(near_plane).cpu().numpy(), torch.max(far_plane).cpu().numpy(), torch.max(h).cpu().numpy(), torch.max(w).cpu().numpy(), intrinsic.cpu().numpy()[0], vox_query=self.opt.NN<0)

            sample_pnt_mask = sample_pidx >= 0
            B, R, SR, K = sample_pidx.shape
            sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()
            sampled_embedding = torch.index_select(torch.cat([self.xyz[None, ...], point_xyz_pers_tensor, self.points_embeding], dim=-1), 1, sample_pidx).view(B, R, SR, K, self.points_embeding.shape[2]+self.xyz.shape[1]*2)

            sampled_color = None if self.points_color is None else torch.index_select(self.points_color, 1, sample_pidx).view(B, R, SR, K, self.points_color.shape[2])

            sampled_dir = None if self.points_dir is None else torch.index_select(self.points_dir, 1, sample_pidx).view(B, R, SR, K, self.points_dir.shape[2])

            sampled_conf = None if self.points_conf is None else torch.index_select(self.points_conf, 1, sample_pidx).view(B, R, SR, K, self.points_conf.shape[2])

            sampled_Rw2c = self.Rw2c if self.Rw2c.dim() == 2 else torch.index_select(self.Rw2c, 0, sample_pidx).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])
            return sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding[..., 6:], sampled_embedding[..., 3:6], sampled_embedding[..., :3], sample_pnt_mask, sample_loc, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, ray_mask_tensor, vsize, self.grid_vox_sz, point_xyz_pers_tensor, raypos_tensor, index_tensor

        #seq_id, vid = inputs["seq_id"].item(), inputs['vid'].item()
        #if vid==1001:
        #    xyz_fov, _, fov_ids,_ = get_lidar_in_image_fov(self.xyz[inputs['seq_id']].squeeze(), inputs["c2w"].squeeze(), intrinsic.squeeze(), xmin=0, ymin=0, xmax=int(w), ymax=int(h), return_more=True, mask=mask)
        #    points_embeding_fov = self.points_embeding[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
        #    points_color_fov = self.points_color[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
        #    points_dir_fov = self.points_dir[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
        #    print(xyz_fov.shape, '-----xyz_fov shape')
        #    np.savetxt('scene0006_chair_'+str(vid)+'.txt', xyz_fov.cpu().numpy())
        #    np.savez('scene0006_chair_for_seqid1.npz', xyz=xyz_fov.cpu().numpy(), embed=points_embeding_fov.squeeze().cpu().numpy(), color=points_color_fov.squeeze().cpu().numpy(), dir=points_dir_fov.squeeze().cpu().numpy())
        #    exit()
        #cprint.info("NeuralPoints xyz{} embed{} color{}".format(len(self.xyz), len(self.points_embeding), len(self.points_color)))
        try:
            self.xyz_fov, _, fov_ids, pts_2d = get_lidar_in_image_fov(self.xyz[inputs['seq_id']].squeeze(), inputs["c2w"].squeeze(), intrinsic.squeeze(), xmin=0, ymin=0, xmax=int(w), ymax=int(h), return_more=True)
            self.points_embeding_fov = self.points_embeding[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
            self.points_color_fov = self.points_color[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
            self.points_dir_fov = self.points_dir[inputs['seq_id']].squeeze(0).squeeze(0)[fov_ids].unsqueeze(0)
        except:
            self.xyz_fov, self.points_embeding_fov, self.points_color_fov, self.points_dir_fov = None, None, None, None
            return

        if self.opt.progressive_distill and self.opt.all_sample_loc==False:
            return

        self.points_conf_fov = None
        sample_pidx, sample_loc, ray_mask_tensor, point_xyz_pers_tensor, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, vsize, raypos_tensor, index_tensor = self.get_point_indices(inputs, camrotc2w, campos, pixel_idx, torch.min(near_plane).cpu().numpy(), torch.max(far_plane).cpu().numpy(), torch.max(h).cpu().numpy(), torch.max(w).cpu().numpy(), intrinsic.cpu().numpy()[0], vox_query=self.opt.NN<0)

        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()
        sampled_embedding = torch.index_select(torch.cat([self.xyz_fov[None, ...], point_xyz_pers_tensor, self.points_embeding_fov], dim=-1), 1, sample_pidx).view(B, R, SR, K, self.points_embeding_fov.shape[2]+self.xyz_fov.shape[1]*2)

        sampled_color = None if self.points_color_fov is None else torch.index_select(self.points_color_fov, 1, sample_pidx).view(B, R, SR, K, self.points_color_fov.shape[2])

        sampled_dir = None if self.points_dir_fov is None else torch.index_select(self.points_dir_fov, 1, sample_pidx).view(B, R, SR, K, self.points_dir_fov.shape[2])

        sampled_conf = None if self.points_conf_fov is None else torch.index_select(self.points_conf_fov, 1, sample_pidx).view(B, R, SR, K, self.points_conf_fov.shape[2])

        sampled_Rw2c = self.Rw2c if self.Rw2c.dim() == 2 else torch.index_select(self.Rw2c, 0, sample_pidx).view(B, R, SR, K, self.Rw2c.shape[1], self.Rw2c.shape[2])
        return sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding[..., 6:], sampled_embedding[..., 3:6], sampled_embedding[..., :3], sample_pnt_mask, sample_loc, sample_loc_w_tensor, sample_ray_dirs_tensor, sample_local_ray_dirs_tensor, ray_mask_tensor, vsize, self.grid_vox_sz, point_xyz_pers_tensor, raypos_tensor, index_tensor
