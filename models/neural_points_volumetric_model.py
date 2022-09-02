import torch
from .base_rendering_model import *
from .neural_points.neural_points import NeuralPoints
from .aggregators.point_aggregators import PointAggregator
import os
from .helpers.networks import positional_encoding
from cprint import *
from pytorch3d.ops import knn_points
class NeuralPointsVolumetricModel(BaseRenderingModel):

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
        if self.opt.progressive_distill or self.opt.all_sample_loc:
            return self.net_ray_marching(**self.input)
        return self.fill_invalid(self.net_ray_marching(**self.input), self.input)

    def fill_invalid(self, output, input):
        # ray_mask:             torch.Size([1, 1024])
        # coarse_is_background: torch.Size([1, 336, 1])  -> 1, 1024, 1
        # coarse_raycolor:      torch.Size([1, 336, 3])  -> 1, 1024, 3
        # coarse_point_opacity: torch.Size([1, 336, 24]) -> 1, 1024, 24
        ray_mask = output["ray_mask"]
        B, OR = ray_mask.shape
        ray_inds = torch.nonzero(ray_mask) # 336, 2

        if self.opt.load_points == 10 and self.opt.all_sample_loc==False:
            decoded_features_tensor = torch.zeros([B, OR, self.opt.SR, 4], dtype=output["decoded_features"].dtype, device=output["decoded_features"].device)
            decoded_features_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output["decoded_features"]

            sample_loc_tensor = torch.zeros([B, OR, self.opt.SR, 3], dtype=output["sample_loc"].dtype, device=output["sample_loc"].device)
            sample_loc_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output["sample_loc"]

            sample_loc_w_tensor = torch.zeros([B, OR, self.opt.SR, 3], dtype=output["sample_loc_w"].dtype, device=output["sample_loc_w"].device)
            sample_loc_w_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output["sample_loc_w"]

            ray_valid_tensor = torch.zeros([B, OR, self.opt.SR], dtype=output["ray_valid"].dtype, device=output["ray_valid"].device)
            ray_valid_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output["ray_valid"]

            output["decoded_features"] = decoded_features_tensor
            output["sample_loc"] = sample_loc_tensor
            output["sample_loc_w"] = sample_loc_w_tensor
            output["ray_valid"] = ray_valid_tensor
            return output

        coarse_is_background_tensor = torch.ones([B, OR, 1], dtype=output["coarse_is_background"].dtype, device=output["coarse_is_background"].device)
        # print("coarse_is_background", output["coarse_is_background"].shape)
        # print("coarse_is_background_tensor", coarse_is_background_tensor.shape)
        # print("ray_inds", ray_inds.shape, ray_mask.shape)
        coarse_is_background_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_is_background"]
        output["coarse_is_background"] = coarse_is_background_tensor
        output['coarse_mask'] = 1 - coarse_is_background_tensor

        if "bg_ray" in self.input:
            coarse_raycolor_tensor = coarse_is_background_tensor * self.input["bg_ray"]
            coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] += output["coarse_raycolor"][0]
        else:
            coarse_raycolor_tensor = self.tonemap_func(
                torch.ones([B, OR, 3], dtype=output["coarse_raycolor"].dtype, device=output["coarse_raycolor"].device) * input["bg_color"][None, ...])
            coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_raycolor"]
        output["coarse_raycolor"] = coarse_raycolor_tensor

        coarse_point_opacity_tensor = torch.zeros([B, OR, output["coarse_point_opacity"].shape[2]], dtype=output["coarse_point_opacity"].dtype, device=output["coarse_point_opacity"].device)
        coarse_point_opacity_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_point_opacity"]
        output["coarse_point_opacity"] = coarse_point_opacity_tensor

        queried_shading_tensor = torch.ones([B, OR, output["queried_shading"].shape[2]], dtype=output["queried_shading"].dtype, device=output["queried_shading"].device)
        queried_shading_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["queried_shading"]
        output["queried_shading"] = queried_shading_tensor

        if self.opt.prob == 1 and "ray_max_shading_opacity" in output:
            # print("ray_inds", ray_inds.shape, torch.sum(output["ray_mask"]))
            output = self.unmask(ray_inds, output, ["ray_max_sample_loc_w", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", "shading_avg_embedding", "ray_max_far_dist"], B, OR)
        return output

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

            net_params = net_params + [par[1] for par in param_lst if not par[0].startswith("neural_points")]
            neural_params = neural_params + [par[1] for par in param_lst if par[0].startswith("neural_points")]

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
        self.l1loss = torch.nn.L1Loss(reduce=False)
        self.bceloss= torch.nn.BCELoss(reduce=False)

        if opt.agg_type == "mlp":
            self.aggregator = aggregator
        elif opt.agg_type == "attention":
            from .perceiver.model.core.modules import PerceiverEncoder, PerceiverDecoder, Encoder
            if opt.attention_type == "normal":
                self.perceiver_encoder = Encoder(num_latents=opt.light_N, \
                                                      num_latent_channels=opt.light_D, \
                                                      num_cross_attention_qk_channels=opt.light_C, \
                                                      num_input_channels=opt.light_C, \
                                                      num_cross_attention_heads=1,
                                                      num_self_attention_heads=opt.light_num_self_attention_heads,
                                                      num_self_attention_layers_per_block=opt.light_num_self_attention_layers_per_block,
                                                      num_self_attention_blocks=opt.light_num_self_attention_blocks,
                                                      dropout=0.0,
                                                      )
            elif opt.attention_type == "perceiver":
                self.perceiver_encoder = PerceiverEncoder(num_latents=opt.light_N, \
                                                      num_latent_channels=opt.light_D, \
                                                      num_cross_attention_qk_channels=opt.light_C, \
                                                      num_input_channels=opt.light_C, \
                                                      num_cross_attention_heads=1,
                                                      num_self_attention_heads=opt.light_num_self_attention_heads,
                                                      num_self_attention_layers_per_block=opt.light_num_self_attention_layers_per_block,
                                                      num_self_attention_blocks=opt.light_num_self_attention_blocks,
                                                      dropout=0.0,
                                                      )
            self.perceiver_decoder = PerceiverDecoder(num_latent_channels=opt.light_D, \
                                                      num_output_query_channels=opt.light_D, \
                                                      #num_output_query_channels=opt.shading_color_channel_num, \
                                                      num_cross_attention_heads=1,
                                                      dropout=0.0,
                                                      perceiver_io_type=opt.perceiver_io_type,
                                                      cat_raydir=True)

        self.num_pos_freqs = num_pos_freqs
        self.num_viewdir_freqs = num_viewdir_freqs

        self.render_func = render_func
        self.blend_func = blend_func

        self.tone_map = tonemap_func
        self.return_depth = is_compute_depth
        self.return_color = True
        self.opt = opt
        self.neural_points = neural_points

    def forward(self,
                campos,
                raydir,
                seq_id=None,
                local_raydir=None,
                c2w=None,
                gt_image=None,
                bg_color=None,
                camrotc2w=None,
                pixel_idx=None,
                near=None,
                far=None,
                focal=None,
                h=None,
                w=None,
                intrinsic=None,
                sample_loc_w_loaded=None,
                sample_loc_loaded=None,
                ray_valid_loaded=None,
                decoded_features_loaded=None,
                vid=None,
                **kargs):
        output = {}
        # B, channel, 292, 24, 32;      B, 3, 294, 24, 32;     B, 294, 24;     B, 291, 2
        weight = None
        if self.opt.progressive_distill and self.opt.all_sample_loc==False:
            self.neural_points({"pixel_idx": pixel_idx, "camrotc2w": camrotc2w, "campos": campos, "near": near, "far": far,"focal": focal, "h": h, "w": w, "intrinsic": intrinsic,"gt_image":gt_image, "raydir":raydir, "c2w":c2w, \
                                "local_raydir":local_raydir, "seq_id":seq_id, "vid":vid})
            ray_valid = ray_valid_loaded
            sample_loc_w = sample_loc_w_loaded
            sample_loc = sample_loc_loaded
            vsize = self.opt.vsize
            sampled_Rw2c = torch.tensor([[1., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 1.]], device='cuda')
            ray_mask_tensor = torch.sum(ray_valid, -1)
            ray_mask_tensor[ray_mask_tensor>1]=1
            ray_mask_tensor = ray_mask_tensor.reshape(1, -1)
        else:
            sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, \
                sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, sample_local_ray_dirs, \
                ray_mask_tensor, vsize, grid_vox_sz, point_xyz_pers_tensor, raypos_tensor, index_tensor = self.neural_points({"pixel_idx": pixel_idx, "camrotc2w": camrotc2w, "campos": campos, "near": near, "far": far, \
                                    "focal": focal, "h": h, "w": w, "intrinsic": intrinsic,"gt_image":gt_image, "raydir":raydir, "c2w":c2w, "local_raydir":local_raydir, "seq_id":seq_id})
            ray_valid = torch.any(sample_pnt_mask, dim=-1)
        output["ray_mask"] = ray_mask_tensor

        query_points = sample_loc_w[ray_valid][None,...].contiguous()
        if self.opt.agg_type=='mlp':
            decoded_features, ray_valid, weight, conf_coefficient = self.aggregator(sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, grid_vox_sz)
            if self.opt.load_points==10 and self.opt.all_sample_loc==False:
                output['sample_loc'] = sample_loc
                output['sample_loc_w'] = sample_loc_w
                output['ray_valid'] = ray_valid
                output['decoded_features'] = decoded_features
                return output
        elif self.opt.agg_type=='attention' and self.neural_points.xyz_fov != None:
            query_points_local = sample_loc[ray_valid][None, ...].contiguous()
            if self.opt.k_type == 'knn':
                dists, assign_index, ref_xyz = knn_points(p1=query_points, p2=self.neural_points.xyz_fov[None, ...], K=self.opt.knn_k, return_nn=True)
                #dists_flag = dists<=self.opt.radius
#                ray_valid_new = torch.any(dists_flag, dim=-1)

                #print(ref_xyz.shape, assign_index.shape, ray_valid_new.shape, torch.sum(ray_valid_new))
                #ref_xyz = ref_xyz[ray_valid_new==True]
                #assign_index = assign_index[ray_valid_new==True]
                #print(ref_xyz.shape, assign_index.shape, ray_valid_new.shape)
                #query_points = query_points[ray_valid_new==True]
                #query_points_local = query_points_local[ray_valid_new==True]
                #exit()

                ref_xyz = ref_xyz.reshape(1, -1, self.opt.knn_k, 3)
                ref_fea = self.neural_points.points_embeding_fov.squeeze()[assign_index.squeeze()].reshape(-1, self.opt.knn_k, self.neural_points.points_embeding_fov.shape[-1])
                if self.opt.embed_color:
                    ref_col = self.neural_points.points_color_fov.squeeze()[assign_index.squeeze()].reshape(-1, self.opt.knn_k, self.neural_points.points_color_fov.shape[-1])
                    ref_dir = self.neural_points.points_dir_fov.squeeze()[assign_index.squeeze()].reshape(-1, self.opt.knn_k, self.neural_points.points_dir_fov.shape[-1])

                if self.opt.clip_knn:
                    dists_flag = (dists <= 0.008).squeeze()
                    dists_flag_sorted, sorted_indices = torch.sort(dists_flag.float(), descending=True, dim=-1)
                    assert (dists_flag_sorted[..., 0]>0).all() == True
                    #pad = torch.gather(input=ref_xyz.squeeze(), dim=1, index=sorted_indices[...,0:1][...,None].repeat(1,1,3))
                    #ref_xyz = torch.where(dists_flag[...,None].repeat(1,1,3), ref_xyz.squeeze(), pad.repeat(1,8,1))
                    ref_xyz = torch.where(dists_flag[...,None], ref_xyz.squeeze(), torch.gather(input=ref_xyz.squeeze(), dim=1, index=sorted_indices[...,0:1][...,None])).reshape(1, -1, self.opt.knn_k, 3)
                    ref_fea = torch.where(dists_flag[...,None], ref_fea, torch.gather(input=ref_fea, dim=1, index=sorted_indices[...,0:1][...,None]))
                    ref_col = torch.where(dists_flag[...,None], ref_col, torch.gather(input=ref_col, dim=1, index=sorted_indices[...,0:1][...,None]))
                    ref_dir = torch.where(dists_flag[...,None], ref_dir, torch.gather(input=ref_dir, dim=1, index=sorted_indices[...,0:1][...,None]))
                ref_xyz_pers = self.w2pers(ref_xyz.view(-1,3), camrotc2w, campos).reshape(1, -1, self.opt.knn_k, 3)
                #  torch_cluster knn
                #assign_index = knn(self.neural_points.xyz_fov.squeeze(), query_points.squeeze().view(-1,3), 8)[1]
                #ref_xyz = self.neural_points.xyz_fov.squeeze()[assign_index].reshape(1, -1, 8, 3)
                #ref_fea = self.neural_points.points_embeding_fov.squeeze()[assign_index].reshape(-1, 8, self.neural_points.points_embeding_fov.shape[-1])
                #ref_xyz_pers = self.w2pers(ref_xyz.view(-1,3), camrotc2w, campos).reshape(1, -1, 8, 3)
            elif self.opt.k_type == 'voxel':
                ref_xyz = sampled_xyz[ray_valid].reshape(1, -1, self.opt.K, 3)
                ref_fea = sampled_embedding[ray_valid].reshape(-1, self.opt.K, sampled_embedding.shape[-1])
                ref_xyz_pers = sampled_xyz_pers[ray_valid].reshape(1, -1, self.opt.K, 3)
                if self.opt.embed_color:
                    ref_col = sampled_color[ray_valid].reshape(-1, self.opt.K, sampled_color.shape[-1])
                    ref_dir = sampled_dir[ray_valid].reshape(-1, self.opt.K, sampled_dir.shape[-1])

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
            if self.opt.embed_color:
                memory = self.perceiver_encoder(torch.cat((ref_fea, dists_flat.view(nq, k, dists_flat.shape[-1]), ref_col, ref_dir), dim=-1))
            else:
                memory = self.perceiver_encoder(torch.cat((ref_fea, dists_flat.view(nq, k, dists_flat.shape[-1])), dim=-1))
            # --- obtain the feature for sample loc ---#
            if self.opt.progressive_distill and self.opt.all_sample_loc==False:
                sample_ray_dirs = raydir.reshape(1,-1,1,3).expand(-1, -1, self.opt.SR, -1).contiguous()
                sample_local_ray_dirs = local_raydir.reshape(1,-1,1,3).expand(-1, -1, self.opt.SR, -1).contiguous()
            else:
                sample_ray_dirs = torch.masked_select(raydir, ray_mask_tensor[..., None]>0).reshape(raydir.shape[0],-1,3)[...,None,:].expand(-1, -1, self.opt.SR, -1).contiguous()
                sample_local_ray_dirs = torch.masked_select(local_raydir, ray_mask_tensor[..., None]>0).reshape(local_raydir.shape[0],-1,3)[...,None,:].expand(-1, -1, self.opt.SR, -1).contiguous()

            if self.opt.ray_dir_type=='local':
                tmp_ray_dir = sample_local_ray_dirs[ray_valid]
            else:
                tmp_ray_dir = sample_ray_dirs[ray_valid]
            tmp_ray_dir = positional_encoding(tmp_ray_dir, self.opt.num_perceiver_io_freqs)[:,None,:]
            if self.opt.perceiver_io_type == 'each_sample_loc':
                query_pcd_fea, query_pcd_alpha = self.perceiver_decoder(memory, nq, tmp_ray_dir)
            else:
                query_pcd_fea, query_pcd_alpha = self.perceiver_decoder(memory, positional_encoding(query_points_local.squeeze(), 6)[:, None,:], tmp_ray_dir)

            decoded_features = torch.zeros([torch.numel(ray_valid), 4], dtype=torch.float32, device=ray_valid.device)
            decoded_features[ray_valid.view(-1)] = torch.cat([query_pcd_alpha, query_pcd_fea], dim=-1).squeeze()
            decoded_features = decoded_features.reshape(1, -1, self.opt.SR, 4)

            ray_valid = ray_valid.reshape(1, -1, self.opt.SR)

        else:
            ref_xyz, ref_xyz_pers, ref_fea, ref_col, ref_dir = None, None, None, None, None
            decoded_features = torch.zeros_like(decoded_features_loaded)

        if self.opt.all_sample_loc:
            '''
            R*24*4 --> R*400*4 --> OR*400*4
            '''
            index_tensor = index_tensor.squeeze().long()[...,None].repeat(1,1,decoded_features.shape[-1])
            decoded_features_all = torch.zeros((index_tensor.shape[0], self.opt.z_depth_dim, decoded_features.shape[-1])).cuda()
            decoded_features_all.scatter_(1, index_tensor, decoded_features.squeeze(0))[...,None]

            B, OR = ray_mask_tensor.shape
            ray_inds = torch.nonzero(ray_mask_tensor) # 336, 2

            decoded_features = torch.zeros([B, OR, 400, 4], dtype=decoded_features.dtype, device=decoded_features.device)
            decoded_features[ray_inds[..., 0], ray_inds[..., 1], ...] = decoded_features_all

            if self.opt.load_points==10:
                output['decoded_features'] = decoded_features
                return output

            raypos_tensor_loc = self.w2pers(raypos_tensor.view(-1,3), camrotc2w, campos).reshape(raypos_tensor.shape)
            ray_dist = torch.cummax(raypos_tensor_loc[..., 2], dim=-1)[0]
        else:
            ray_dist = torch.cummax(sample_loc[..., 2], dim=-1)[0]

        #==================== ray dist ==================#
        ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)], dim=-1)
        mask = ray_dist < 1e-8
        if self.opt.raydist_mode_unit > 0:
            mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
        mask = mask.to(torch.float32)
        ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
        ray_dist *= ray_valid.float()

        output['ray_dist'] = ray_dist
        output["queried_shading"] = torch.logical_not(torch.any(ray_valid, dim=-1, keepdims=True)).repeat(1, 1, 3).to(torch.float32)

        if self.return_color:
            if "bg_ray" in kargs:
                bg_color = None
            (
                ray_color,
                point_color,
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _, opacity_gt,
            ) = ray_march(ray_dist, ray_valid, decoded_features, self.render_func, self.blend_func, bg_color, decoded_features_loaded)
            ray_color = self.tone_map(ray_color)
            output["coarse_raycolor"] = ray_color
            output["coarse_point_opacity"] = opacity
        else:
            (
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = alpha_ray_march(ray_dist, ray_valid, decoded_features, self.blend_func)

        if self.opt.progressive_distill:
            if self.opt.all_sample_loc==False:
                opacity_gt = opacity_gt[ray_valid][...,None]
                opacity    = opacity[ray_valid][..., None]

                #rgb    = query_pcd_fea.squeeze()
                rgb    = decoded_features[...,1:][ray_valid]
                rgb_gt = decoded_features_loaded[...,1:][ray_valid]

                opacity_loss = self.l1loss(opacity, opacity_gt)
                rgb_loss = torch.mean(self.l1loss(rgb, rgb_gt), dim=-1, keepdim=True)
            else:
                opacity_loss = self.l1loss(opacity[...,None], opacity_gt[...,None])
                rgb_loss = torch.mean(self.l1loss(decoded_features[...,1:], decoded_features_loaded[...,1:]), dim=-1, keepdim=True)
            loss = opacity_loss + rgb_loss
            if self.opt.relative_thresh:
                failed_index = loss > torch.max(loss).item()*self.opt.prob_thresh
            else:
                failed_index = loss > self.opt.prob_thresh
            if self.opt.all_sample_loc:
                failed_sample_loc = raypos_tensor.view(-1,3)[torch.flatten(failed_index)]
            else:
                failed_sample_loc = query_points.squeeze()[failed_index.squeeze()]
            output['failed_sample_loc'] = failed_sample_loc
            output['loss_alpha'] = torch.mean(opacity_loss)
            output['loss_rgb'] = torch.mean(rgb_loss)

        if self.return_depth:
            alpha_blend_weight = opacity * acc_transmission
            weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
            avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
            output["coarse_depth"] = avg_depth
        output["coarse_is_background"] = background_transmission
        if weight is not None:
            output["weight"] = weight.detach()
            output["blend_weight"] = blend_weight.detach()
            output["conf_coefficient"] = conf_coefficient
        return output

    def w2pers(self, point_xyz, camrotc2w, campos):
        point_xyz_shift = point_xyz[None, ...] - campos[:, None, :]
        xyz = torch.sum(camrotc2w[:, None, :, :] * point_xyz_shift[:, :, :, None], dim=-2)
        xper = xyz[:, :, 0] / xyz[:, :, 2]
        yper = xyz[:, :, 1] / xyz[:, :, 2]
        return torch.stack([xper, yper, xyz[:, :, 2]], dim=-1)
