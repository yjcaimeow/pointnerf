import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from neural_render.neural_renderer import NeuralRenderer
#from models.point_pe import point_pe

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def trans_world2nerf(global_frame_points, pose_nerf2world):
    pose_world2nerf = torch.linalg.inv(pose_nerf2world)
    point_at_nerf_frame = pose_world2nerf[:3,:3] @ global_frame_points.T + pose_world2nerf[:3, 3][:, None]
    return point_at_nerf_frame.T

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class MipEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x[:,:d]) # append mean
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        min_freq = self.kwargs['min_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands_y = 2.**torch.linspace(min_freq, max_freq, steps=N_freqs)
            freq_bands_w = 4.**torch.linspace(min_freq, max_freq, steps=N_freqs)
        else:
            freq_bands_y = torch.linspace(2.**min_freq, 2.**max_freq, steps=N_freqs)
            freq_bands_w = torch.linspace(4.**min_freq, 4.**max_freq, steps=N_freqs)

        for ctr in range(len(freq_bands_y)):
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda inputs, p_fn=p_fn, freq_y=freq_bands_y[ctr], freq_w=freq_bands_w[ctr] : p_fn(inputs[:,:d] * freq_y) * torch.exp((-0.5) * freq_w * inputs[:,d:]))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def get_mip_embedder(multires, min_multires=0, i=0, include_input=True):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : include_input,
                'input_dims' : 3,
                'min_freq_log2': min_multires,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = MipEmbedder(**embed_kwargs)
    embed = lambda inputs, eo=embedder_obj : eo.embed(inputs)
    return embed, embedder_obj.out_dim

class Attention(nn.Module):
    def __init__(self, args, dim=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = args.num_heads
        head_dim = dim // args.num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.K = args.close_points_number
        #self.close_points_number = 8
    def forward(self, q, k, v):
        #print (q.shape, k.shape, v.shape)
        #import pdb;pdb.set_trace()
        #B, N, C = x.shape
        #k = k.reshape(-1, self.close_points_number, self.dim)
        #q = q.reshape(-1, self.close_points_number, self.dim)
        #BHW, K, _ = k.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.q_linear(q) # bhw, d
        q = q.reshape(q.shape[0], self.num_heads, q.shape[-1] // self.num_heads)

        k = self.k_linear(k) # bhwk, d
        k = k.reshape(-1, self.K, self.num_heads, k.shape[-1] // self.num_heads)

        v = self.v_linear(v) # bhwk, d
        v = v.reshape(-1, self.K, self.num_heads, v.shape[-1] // self.num_heads)

        #print ('processed qkv shape', q.shape, k.shape, v.shape)

        #attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.einsum('ijm,ikjm->ijk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #print (attn.shape, v.shape, '--- attention final')
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = torch.einsum('ijk,ikjm->ijm', attn @ v).reshape(B,-1)
        x = torch.einsum('ijk,ikjm->ijm', attn, v).reshape(-1, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args, output_ch=3, height=160, width=240):
        """
        """
        super(MultiHeadAttention, self).__init__()
        self.K = args.close_points_number
        self.skips = [4]
        self.embed_dir = args.embed_dir
        self.render_type = args.render_type
        if self.embed_dir:
            input_ch = args.net_ch + 27
            input_ch_views = 27
        else:
            input_ch = args.net_ch + 3
            input_ch_views = 3
        self.input_ch = input_ch

        self.enhance_feature = args.enhance_feature

        if args.enhance_feature:
            from models.transformer import TransformerEncoderLayer
            self.transformer_enhance_feature = TransformerEncoderLayer(d_model=args.bottleneck_size, \
                    dim_feedforward=args.bottleneck_size, nhead=2, normalize_before=args.normalize_before)

            self.max_steps = 100
            self.fn = nn.Linear(args.bottleneck_size+27, args.bottleneck_size)
            self.fn_thetaV = nn.Linear(args.bottleneck_size, args.net_ch)
            self.fn_thetaK = nn.Linear(args.bottleneck_size, args.net_ch)
        else:
            self.fn_thetaV = nn.Linear(args.bottleneck_size+27, args.net_ch)
            self.fn_thetaK = nn.Linear(args.bottleneck_size+27, args.net_ch)

        self.fn_thetaQ = nn.Linear(27, args.net_ch)
        self.multi_head_attention = Attention(args=args, dim=args.net_ch)

        if self.render_type=='cnn':
            self.neural_render_2d = NeuralRenderer(input_dim=input_ch)
        elif self.render_type=='mlp':
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch, args.netwidth)] + [nn.Linear(args.netwidth, args.netwidth) if i not in self.skips \
                        else nn.Linear(args.netwidth + input_ch, args.netwidth) for i in range(args.netdepth-2)])
            self.output_linear = nn.Linear(args.netwidth, output_ch)
        elif self.render_type == 'nerf':
            self.neural_render = NeRF(input_ch = args.net_ch, input_ch_views=input_ch_views, use_viewdirs=True)
        elif self.render_type == 'stylegenerator':
            from networks.networks_stylegan2 import Generator
            self.z_dim  = 512
            self.seed = 0
            self.neural_render_2d = Generator(z_dim=self.z_dim, c_dim=0, w_dim=512, img_resolution=512, img_channels=3)

    def forward(self,rays, points, directions, point_cloud=None, height=160, width=240, bs=1, z=None):
        """
        rays    : BHW  , 27
        points  : BHWK , bottleneck_size+27
        position: BHWK , 3
        """
        rays = rays.to(torch.float32)
        points = points.to(torch.float32)
        if self.enhance_feature:
            points = self.fn(points) # pre_processed_k
            low = point_cloud.min(0).values
            high = point_cloud.max(0).values
            steps = high - low
            steps *= self.max_steps / steps.max()
            peed_position = point_pe(point_cloud)

            points = points.reshape(-1, self.K, points.shape[-1]).permute((1,0,2))
            peed_position = peed_position.reshape(-1, self.K, peed_position.shape[-1]).permute((1,0,2))
            points = self.transformer_enhance_feature(src=points, pos=peed_position).permute((1,0,2))
            points = points.reshape(-1, points.shape[-1])
        ppd_k = self.fn_thetaK(points) # pre_processed_k
        ppd_v = self.fn_thetaV(points) # pre_processed_v
        ppd_q = self.fn_thetaQ(rays)   # pre_processed_q
        rgb_feature_map = self.multi_head_attention(ppd_q,ppd_k,ppd_v)
        if self.embed_dir:
            input_pts = torch.cat((rgb_feature_map, rays),-1)
        else:
            input_pts = torch.cat((rgb_feature_map, directions),-1)
        if self.render_type=='cnn':
            return self.neural_render_2d(input_pts.reshape(-1, height, width, self.input_ch))
        elif self.render_type == 'stylegenerator':
            return self.neural_render_2d(z=z, feature_map = input_pts.reshape(-1, height, width, self.input_ch).permute(0,3,1,2))
        elif self.render_type=='mlp':
            h = input_pts
            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)
            return outputs.reshape(-1, height, width, 3)
        elif self.render_type == 'nerf':
            raw = self.neural_render(input_pts)
            rgb = torch.sigmoid(raw[...,:3]).reshape(-1, height, width, 3)
            return rgb

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_radii_for_test(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[np.newaxis, ..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    dx = torch.sqrt(
        torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii

def get_coors_np(H, W, K, c2w):
    f, cx, cy = K[0,0], K[0,2], K[1,2]
    z_near, z_far = 0., 100.
    corners = np.array([[0.,0.], [W-1, 0.], [0, H-1], [W-1, H-1]], dtype=np.float32)
    polygon = []
    for corner in corners:
        u, v = corner
        xc_near, yc_near, zc_near = (u-cx)*z_near/f, -(v-cy)*z_near/f, -z_near
        xc_far,  yc_far,  zc_far  = (u-cx)*z_far/f, -(v-cy)*z_far/f, -z_far
        polygon.append([xc_near, yc_near, zc_near])
        polygon.append([xc_far, yc_far, zc_far])
    polygon_camera = np.asarray(polygon)
    polygon_world = c2w[:3,:3] @ polygon_camera.T + c2w[:3, 3][:, None]
    return polygon_world.T

#def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
def get_dtu_raydir(height, width, intrinsic, c2w):
    rot = c2w[0:3, 0:3]
    # rot is c2w
    ## pixelcoords: H x W x 2
    px, py = np.meshgrid(np.arange(0, width).astype(np.float32),
                         np.arange(0, height).astype(np.float32))
    pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)
    x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1)
    dirs2 = np.sum(dirs[...,None,:] * rot[:,:], axis=-1)
    rays_d = dirs @ rot[:,:].T #
    print (rays_d)
    print ('=========')
    print ((rays_d==dirs2).any())
    exit()
#    if dir_norm:
        # print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
#        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)

    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w, returndir=False):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #print (c2w, '----get_rays_np')
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    if returndir:
        return rays_o, rays_d, dirs
    else:
        return rays_o, rays_d

'''
def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:,0]-W*0.5)/focal, -(coords[:,1]-H*0.5)/focal
    dirs = np.stack([i,j,-np.ones_like(i)],-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d
'''

def get_rays_by_coord_np(H, W, K, c2w, coords):
    #i, j = (coords[:,0]-W*0.5)/focal, -(coords[:,1]-H*0.5)/focal
    i, j = (coords[:,0]-K[0][2])/K[0][0], -(coords[:,1]-K[1][2])/K[1][1]
    dirs = np.stack([i,j,-np.ones_like(i)],-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins.

    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        bins: jnp.ndarray(float32), [batch_size, num_bins + 1].
        weights: jnp.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, axis=-1, keepdims=True)
    padding = torch.maximum(torch.tensor(0), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.minimum(torch.tensor(1), torch.cumsum(pdf[..., :-1], axis=-1))

    cdf = torch.cat([
            torch.zeros(list(cdf.shape[:-1]) + [1]), cdf,
            torch.ones(list(cdf.shape[:-1]) + [1])
    ], axis=-1) # [1024, 65]

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = np.arange(num_samples) * s
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
        jitter = np.random.uniform(high=s - np.finfo('float32').eps, size=list(cdf.shape[:-1]) + [num_samples])
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = u + jitter
        u = np.minimum(u, 1. - np.finfo('float32').eps)
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = np.linspace(0., 1. - np.finfo('float32').eps, num_samples)
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    u = torch.from_numpy(u).to(cdf)
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), dim=-2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), dim=-2)[0]
        return x0, x1


    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = (u - cdf_g0) / (cdf_g1 - cdf_g0)
    t[t != t] = 0
    t = torch.clamp(t, 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples
