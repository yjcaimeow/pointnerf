from __future__ import print_function
import torch
from pytorch3d.ops import ball_query
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid
from os.path import join
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import itertools

def add_flour(failed_sample_loc, candidates=None, gap=0.2, radius=0.2):
    if candidates is None:
        center = torch.tensor([3.7269, 3.4063, 1.2413]).cuda()
        whl = torch.tensor([8.2886, 8.1767, 3.0916]).cuda()
        range_min,range_max = center-whl/2, center+whl/2

        xs = torch.arange(range_min[0], range_max[0], gap, device='cuda')
        ys = torch.arange(range_min[1], range_max[1], gap, device='cuda')
        zs = torch.arange(range_min[2], range_max[2], gap, device='cuda')

        candidates = torch.cartesian_prod(xs, ys, zs)
    else:
        our_list = [-1,0,1]
        lists = []
        for item in itertools.product(our_list, our_list, our_list):
            lists.append(item)
        lists.remove((0,0,0))

        neighbor_pcds = torch.tensor(lists.remove((0,0,0))).cuda()

        new_candidates = []
        for neighbor_pcd in neighbor_pcds:
            pcd = candidates + (neighbor_pcd * gap).reshape(1,3)
            new_candidates.append(pcd)
        candidates = torch.cat(new_candidates)
        print (candidates.shape, '-------candidates shape')

    idx = ball_query(candidates[None,...], failed_sample_loc[None,...], K=1, radius=radius).idx.squeeze() #N*p1*1

    return candidates[idx>0]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_property2dict(target_dict, object, props):
    for prop in props:
        target_dict[prop] = getattr(object, prop)


def normalize(v, axis=0):
    # axis = 0, normalize each col
    # axis = 1, normalize each row
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def gen_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def unique_lst(list1):
    x = np.array(list1)
    return np.unique(x)
