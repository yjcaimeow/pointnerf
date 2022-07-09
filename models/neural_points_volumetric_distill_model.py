from .base_rendering_model import *
from .neural_points.neural_points import NeuralPoints
from .aggregators.point_aggregators import PointAggregator
import os
from utils import format as fmt
import random
from utils.kitti_object import trans_world2nerf
from .helpers.networks import init_seq, positional_encoding, effective_range
import torch
from torch_cluster import knn
from cprint import *
import time

def to_column_major_torch(x):
    if hasattr(torch, 'contiguous_format'):
        return x.t().clone(memory_format=torch.contiguous_format).t()
    else:
        return x.t().clone().t()

class NeuralPointsVolumetricDistillModel(BaseRenderingModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        BaseRenderingModel.modify_commandline_options(parser, is_train)
        NeuralPoints.modify_commandline_options(parser, is_train)
        PointAggregator.modify_commandline_options(parser, is_train)

        parser.add_argument(
            '--neural_point_dir',
            type=str,
            default=None,
            help='alternative loading neural_point directory')

        parser.add_argument(
            '--embedding_size',
            type=int,
            default=-1,
            help='number of dimensions for latent code embedding')
        parser.add_argument(
            "--loss_embedding_l2_weight",
            type=float,
            default=-1,
            help="weight for the embedding l2 loss",
        )
        parser.add_argument('--loss_kld_weight',
                            type=float,
                            default=-1,
                            help='weight for the VAE kld')

        # encoder
        parser.add_argument(
            "--compute_depth",
            type=int,
            default=0,
            help=
            "If compute detph or not. If false, depth is only computed when depth is required by losses",
        )


        parser.add_argument(
            "--raydist_mode_unit",
            type=int,
            default=0,
            help="if set raydist max as one voxel",
        )

        parser.add_argument(
            '--save_point_freq',
            type=int,
            default=100000,
            help='frequency of showing training results on console')

        parser.add_argument(
            '--alter_step',
            type=int,
            default=0,
            help='0 for no alter,')

        parser.add_argument(
            '--prob',
            type=int,
            default=0,
            help='will be set as 0 for normal traing and 1 for prob, ')


    def add_default_color_losses(self, opt):
        if "coarse_raycolor" not in opt.color_loss_items:
            opt.color_loss_items.append('coarse_raycolor')
        if opt.fine_sample_num > 0:
            opt.color_loss_items.append('fine_raycolor')

    def add_default_visual_items(self, opt):
        opt.visual_items = ['gt_image', 'coarse_raycolor', 'queried_shading']
        if opt.fine_sample_num > 0:
            opt.visual_items.append('fine_raycolor')

    def run_network_models(self):
        res = self.net_ray_marching(**self.input)
        #res =self.fill_invalid(mid, self.input)
        #for key in res.keys():
        #    print (key, res[key].shape)
        #exit()
        return res
        #return self.fill_invalid(self.net_ray_marching(**self.input), self.input)

    # def fill_invalid(self, output, input):
    #     # ray_mask:             torch.Size([1, 1024])
    #     # coarse_is_background: torch.Size([1, 336, 1])  -> 1, 1024, 1
    #     # coarse_raycolor:      torch.Size([1, 336, 3])  -> 1, 1024, 3
    #     # coarse_point_opacity: torch.Size([1, 336, 24]) -> 1, 1024, 24
    #     ray_mask = output["ray_mask"]
    #     B, OR = ray_mask.shape
    #     ray_inds = torch.nonzero(ray_mask) # 336, 2
    #     coarse_is_background_tensor = torch.ones([B, OR, 1], dtype=output["coarse_is_background"].dtype, device=output["coarse_is_background"].device)
    #     # print("coarse_is_background", output["coarse_is_background"].shape)
    #     # print("coarse_is_background_tensor", coarse_is_background_tensor.shape)
    #     # print("ray_inds", ray_inds.shape, ray_mask.shape)
    #     coarse_is_background_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_is_background"]
    #     output["coarse_is_background"] = coarse_is_background_tensor
    #     output['coarse_mask'] = 1 - coarse_is_background_tensor
    #     print ("bg_ray" in self.input)
    #     print (input["bg_color"])
    #     exit()
    #     if "bg_ray" in self.input:
    #         coarse_raycolor_tensor = coarse_is_background_tensor * self.input["bg_ray"]
    #         coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] += output["coarse_raycolor"][0]
    #     else:
    #         coarse_raycolor_tensor = self.tonemap_func(
    #             torch.ones([B, OR, 3], dtype=output["coarse_raycolor"].dtype, device=output["coarse_raycolor"].device) * input["bg_color"][None, ...])
    #         coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_raycolor"]
    #     output["coarse_raycolor"] = coarse_raycolor_tensor

    #     coarse_point_opacity_tensor = torch.zeros([B, OR, output["coarse_point_opacity"].shape[2]], dtype=output["coarse_point_opacity"].dtype, device=output["coarse_point_opacity"].device)
    #     coarse_point_opacity_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_point_opacity"]
    #     output["coarse_point_opacity"] = coarse_point_opacity_tensor

    #     queried_shading_tensor = torch.ones([B, OR, output["queried_shading"].shape[2]], dtype=output["queried_shading"].dtype, device=output["queried_shading"].device)
    #     queried_shading_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["queried_shading"]
    #     output["queried_shading"] = queried_shading_tensor

    #     if self.opt.prob == 1 and "ray_max_shading_opacity" in output:
    #         # print("ray_inds", ray_inds.shape, torch.sum(output["ray_mask"]))
    #         output = self.unmask(ray_inds, output, ["ray_max_sample_loc_w", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", "shading_avg_embedding", "ray_max_far_dist"], B, OR)
    #     return output

    def unmask(self, ray_inds, output, names, B, OR):
        for name in names:
            if output[name] is not None:
                name_tensor = torch.zeros([B, OR, *output[name].shape[2:]], dtype=output[name].dtype, device=output[name].device)
                name_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output[name]
                output[name] = name_tensor
        return output

    def get_additional_network_params(self, opt):
        param = {}
        # additional parameters

        self.aggregator = self.check_getAggregator(opt)
        self.is_compute_depth = opt.compute_depth or not not opt.depth_loss_items
        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, '{}_net_ray_marching.pth'.format(opt.resume_iter))
        checkpoint_path = checkpoint_path if os.path.isfile(checkpoint_path) else None
        if opt.num_point > 0:
            self.neural_points = NeuralPoints(opt.point_features_dim, opt.num_point, opt, self.device, checkpoint=checkpoint_path, feature_init_method=opt.feature_init_method, reg_weight=0., feedforward=opt.feedforward)
        else:
            self.neural_points = None

        add_property2dict(param, self, [
            'aggregator', 'is_compute_depth', "neural_points", "opt"
        ])
        add_property2dict(param, opt, [
            'num_pos_freqs', 'num_viewdir_freqs'
        ])

        return param

    def create_network_models(self, opt):

        params = self.get_additional_network_params(opt)
        # network
        self.net_ray_marching = NeuralPointsRayMarching(
            **params, **self.found_funcs)

        self.model_names = ['ray_marching'] if getattr(self, "model_names", None) is None else self.model_names + ['ray_marching']

        #self.net_ray_marching.to(self.device)
        # parallel
        #if self.opt.gpu_ids:
        #    self.net_ray_marching.to(self.device)
        #    self.net_ray_marching = torch.nn.DataParallel(
        #        self.net_ray_marching, self.opt.gpu_ids)


    def check_getAggregator(self, opt, **kwargs):
        aggregator = PointAggregator(opt)
        return aggregator


    def setup_optimizer(self, opt):
        '''
            Setup the optimizers for all networks.
            This assumes network modules have been added to self.model_names
            By default, it uses an adam optimizer for all parameters.
        '''
        net_params = []
        neural_params = []
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            param_lst = list(net.named_parameters())

            net_params = net_params + [par[1] for par in param_lst if not par[0].startswith("module.neural_points")]
            neural_params = neural_params + [par[1] for par in param_lst if par[0].startswith("module.neural_points")]

        self.net_params = net_params
        self.neural_params = neural_params

        # opt.lr=0
        self.optimizer = torch.optim.Adam(net_params,
                                          lr=opt.lr,
                                          betas=(0.9, 0.999))
        self.neural_point_optimizer = torch.optim.Adam(neural_params,
                                          lr=opt.lr, #/ 5.0,
                                          betas=(0.9, 0.999))
        self.optimizers = [self.optimizer, self.neural_point_optimizer]

    def backward(self, iters):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        if self.opt.is_train:
            self.loss_total.backward()
            if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 0:
                self.optimizer.step()
            if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 1:
                self.neural_point_optimizer.step()


    def optimize_parameters(self, backward=True, total_steps=0):
        self.forward()
        #self.update_rank_ray_miss(total_steps)
        self.backward(total_steps)

    def update_rank_ray_miss(self, total_steps):
        raise NotImplementedError

class NeuralPointsRayMarching(nn.Module):
    def __init__(self,
             tonemap_func=None,
             render_func=None,
             blend_func=None,
             aggregator=None,
             is_compute_depth=False,
             neural_points=None,
             opt=None,
             num_pos_freqs=0,
             num_viewdir_freqs=0,
             **kwargs):
        super(NeuralPointsRayMarching, self).__init__()

        #self.aggregator = aggregator

        self.num_pos_freqs = num_pos_freqs
        self.num_viewdir_freqs = num_viewdir_freqs
        # ray generation

        self.render_func = render_func
        self.blend_func = blend_func

        self.tone_map = tonemap_func
        self.return_depth = is_compute_depth
        self.return_color = True
        self.opt = opt
        self.neural_points = neural_points
        #if self.opt.neural_render=='cnn':
        #    from models.neural_render.neural_renderer import NeuralRenderer
        #    self.neural_render_2d = NeuralRenderer(input_dim=self.opt.shading_color_channel_num)
        #elif self.opt.neural_render=='style':
        #    from models.neural_render.stylegan2_pytorch_8x import StyleVectorizer, Generator
        #    self.S = StyleVectorizer(emb=self.opt.z_dim, depth=8)
        #    self.G = Generator(image_size=512, latent_dim=self.opt.z_dim, network_capacity=self.opt.network_capacity, input_dim=self.opt.shading_color_channel_num)

        if self.opt.perceiver_io:
            from perceiver.model.core.modules import PerceiverEncoder, PerceiverDecoder
            self.perceiver_encoder = PerceiverEncoder(num_latents=self.opt.N, \
                                                      num_latent_channels=self.opt.D, \
                                                      num_cross_attention_qk_channels=self.opt.C, \
                                                      num_input_channels=self.opt.C, \
                                                      num_cross_attention_heads=1,
                                                      num_self_attention_heads=self.opt.num_self_attention_heads,
                                                      num_self_attention_layers_per_block=self.opt.num_self_attention_layers_per_block,
                                                      num_self_attention_blocks=self.opt.num_self_attention_blocks,
                                                      dropout=0.0,
                                                      )
            self.perceiver_decoder = PerceiverDecoder(num_latent_channels=self.opt.D, \
                                                      num_output_query_channels=self.opt.E, \
                                                      num_cross_attention_heads=1,
                                                      dropout=0.0,
                                                      perceiver_io_type=self.opt.perceiver_io_type,
                                                      )
        if self.opt.basic_agg == 'attention':
            from perceiver.model.core.modules import PerceiverEncoder, PerceiverDecoder
            self.perceiver_encoder = PerceiverEncoder(num_latents=self.opt.light_N, \
                                                      num_latent_channels=self.opt.light_D, \
                                                      num_cross_attention_qk_channels=self.opt.light_C, \
                                                      num_input_channels=self.opt.light_C, \
                                                      num_cross_attention_heads=1,
                                                      num_self_attention_heads=self.opt.light_num_self_attention_heads,
                                                      num_self_attention_layers_per_block=self.opt.light_num_self_attention_layers_per_block,
                                                      num_self_attention_blocks=self.opt.light_num_self_attention_blocks,
                                                      dropout=0.0,
                                                      )
            self.perceiver_decoder = PerceiverDecoder(num_latent_channels=self.opt.light_D, \
                                                      num_output_query_channels=self.opt.shading_color_channel_num, \
#                                                      num_output_query_channels=self.opt.light_E, \
                                                      num_cross_attention_heads=1,
                                                      dropout=0.0,
                                                      perceiver_io_type="each_sample_loc",
                                                      cat_raydir=self.opt.cat_raydir,
                                                      )
        self.local_rank = int(os.environ["LOCAL_RANK"])
    def latent_to_w(self, style_vectorizer, latent_descr):
        return [(style_vectorizer(latent_descr.cuda()), 8)]

    def styles_def_to_tensor(self, styles_def):
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

    def make_noise(self, batch, latent_dim, n_noise, device):
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
        return noises


    def mixing_noise(self, batch, latent_dim, prob, device='cuda'):
        if prob > 0 and random.random() < prob:
            return self.make_noise(batch, latent_dim, 2, device)
        else:
            return [self.make_noise(batch, latent_dim, 1, device)]

    def forward(self,
                sample_loc,
                sample_loc_w,
                decoded_features,
                ray_dist,
                ray_valid,
                campos,
                local_raydir=None,
                raydir_wonorm=None,
                gt_image=None,
                gt_image_1over8=None,
                bg_color=None,
                camrotc2w=None,
                pixel_idx=None,
                c2w=None,
                near=None,
                far=None,
                focal=None,
                h=None,
                w=None,
                id=None,
                id_in_seq=None,
                seq_id=None,
                intrinsic=None,
                sequence_length_list=None,
                train_sequence_length_list=None,
                **kargs):
        output = {}
        vsize = self.opt.vsize
        height, width = 480, 640
        sample_loc = sample_loc.reshape(1, height*width, 24, 3)
        sample_loc_w = sample_loc_w.reshape(1, height*width, 24, 3)
        ray_valid = ray_valid.reshape(1, height*width, -1)
        decoded_features = decoded_features.reshape(1, height*width, 24, 4)
        #ray_dist_gt = ray_dist.reshape(1, height*width, -1)
        self.neural_points({"h": h, "w": w, "intrinsic": intrinsic, "c2w":c2w})

        query_points = sample_loc_w[ray_valid][None,...].contiguous()
        query_points_local = self.w2pers(query_points, camrotc2w, campos)[None, ...]
        assign_index = knn(self.neural_points.xyz.squeeze(), query_points.squeeze().view(-1,3), 8)[1]
        ##--- get the pcd and feature for each sample loc and knn pcd
        ref_xyz = self.neural_points.xyz.squeeze()[assign_index].reshape(1, -1, 8, 3)
        ref_fea = self.neural_points.points_embeding.squeeze()[assign_index].reshape(-1, 8, self.neural_points.points_embeding.shape[-1])
        ref_xyz_pers = self.w2pers(ref_xyz.view(-1,3), camrotc2w, campos).reshape(1, -1, 8, 3)
        xdist = ref_xyz_pers[..., 0] * ref_xyz_pers[..., 2] - query_points_local[:, :, None, 0] * query_points_local[:, :, None, 2]
        ydist = ref_xyz_pers[..., 1] * ref_xyz_pers[..., 2] - query_points_local[:, :, None, 1] * query_points_local[:, :, None, 2]
        zdist = ref_xyz_pers[..., 2] - query_points_local[:, :, None, 2]
        dists = torch.stack([xdist, ydist, zdist], dim=-1)
        dists = torch.cat([ref_xyz - query_points[..., None, :], dists], dim=-1)
        B, nq, k, _ = dists.shape

        dists_flat = dists.view(-1, dists.shape[-1])
        dists_flat /= (1.0 if self.opt.dist_xyz_deno == 0. else float(self.opt.dist_xyz_deno * np.linalg.norm(vsize)))

        uni_w2c = sampled_Rw2c.dim() == 2
        dists_flat[..., :3] = dists_flat[..., :3] @ sampled_Rw2c if uni_w2c else (dists_flat[..., None, :3] @ sampled_Rw2c).squeeze(-2)
        dists_flat = positional_encoding(dists_flat, self.opt.dist_xyz_freq)
        # --- memory position encoding for the neighbors ----#
        ref_fea = torch.cat([ref_fea, positional_encoding(ref_fea, self.opt.num_feat_freqs)], dim=-1)

        memory = self.perceiver_encoder(torch.cat((ref_fea, dists_flat.view(nq, k, dists_flat.shape[-1])), dim=-1))
        # --- obtain the feature for sample loc ---#
        #tmp_ray_dir = sample_local_ray_dirs_i[ray_valid_i]
        #tmp_ray_dir = positional_encoding(tmp_ray_dir, self.opt.num_perceiver_io_freqs)[:,None,:]
        #print (tmp_ray_dir.shape)
        #exit()
        #query_pcd_fea, query_pcd_alpha = self.perceiver_decoder(memory, nq, tmp_ray_dir)
        query_pcd_fea, query_pcd_alpha = self.perceiver_decoder(memory, nq)

        query_pcd_fea_all = torch.zeros([B * height*width * self.opt.SR , query_pcd_fea.shape[-1]], dtype=torch.float32, device=query_pcd_fea.device)
        query_pcd_alpha_all = torch.zeros([B * height*width * self.opt.SR , 1], dtype=torch.float32, device=query_pcd_fea.device)

        query_pcd_fea_all[ray_valid_i.view(-1)] = query_pcd_fea.squeeze(1)
        query_pcd_alpha_all[ray_valid_i.view(-1)] = query_pcd_alpha.squeeze(1)

        query_pcd_fea_all = query_pcd_fea_all.reshape(B, height*width, self.opt.SR, -1)
        query_pcd_alpha_all = query_pcd_alpha_all.reshape(B, height*width, self.opt.SR, -1)

        #---------------------------------------#
        #---------- prepare ray dist -----------#
        #---------------------------------------#
        ray_dist = torch.cummax(sample_loc[..., 2], dim=-1)[0]
        ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)], dim=-1)

        mask = ray_dist < 1e-8
        if self.opt.raydist_mode_unit > 0:
            mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
        mask = mask.to(torch.float32)
        ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
        ray_dist *= ray_valid.float()

        #---------------------------------------#
        #---------- ray rendering    -----------#
        #---------------------------------------#
        if "bg_ray" in kargs:
            bg_color = None
        (
            ray_color,
            point_color,
            opacity,
            acc_transmission,
            blend_weight,
            background_transmission,
            _,
        ) = ray_march(ray_dist, ray_valid, decoded_features, self.render_func, self.blend_func, bg_color)
        ray_color = self.tone_map(ray_color)
        output["final_coarse_raycolor"] = ray_color

        return output

    def w2pers(self, point_xyz, camrotc2w, campos):
        #----- torch.Size([105261, 3]) torch.Size([1, 3, 3]) torch.Size([1, 3])
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        # print(xyz.shape, (point_xyz_shift[:, None, :] * camrot.T).shape)
        xper = xyz[:, :, 0] / xyz[:, :, 2]
        yper = xyz[:, :, 1] / xyz[:, :, 2]
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)

    def raw2outputs(self, raw, z_vals, rays_d, mask, raw_noise_std=0, white_bkgd=False, pytest=False):
        print (raw.device, 'raw [num_rays, num_samples along ray, 4]')
        print (z_vals.device, 'z_vals: [num_rays, num_samples along ray]')
        print (rays_d.device, 'rays_d: [num_rays, 3]')
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
    #    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        raw2alpha = lambda raw, dists, act_fn=F.softplus: 1.-torch.exp(-act_fn(raw-1)*dists)
        z_vals = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).cuda()], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = (1+2*0.001)/(1+torch.exp(-raw[...,:3]))-0.001
    #    rgb = torch.sigmoid(raw[...,:3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights *= mask
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map
