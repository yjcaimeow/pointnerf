''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import torch
from cprint import *
import time
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert(idx<self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert(idx<self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training')
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass

class kitti_object_video(object):
    ''' Load data for KITTI videos '''
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename) \
            for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename) \
            for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        #assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples)
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert(idx<self.num_samples)
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib

def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(\
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        raw_input()
        pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
        draw_lidar(pc)
        raw_input()
    return

def show_image_with_boxes(img, objects, calib, show3d=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type=='DontCare':continue
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    Image.fromarray(img1).show()
    if show3d:
        Image.fromarray(img2).show()


def trans_world2nerf(global_frame_points, pose_nerf2world):
#    import pdb; pdb.set_trace()
    pose_world2nerf = torch.linalg.inv(pose_nerf2world)
    #point_at_nerf_frame = pose_world2nerf[:3,:3] @ global_frame_points.T + pose_world2nerf[:3, 3][:, None]
    #cprint.err("pose_world2nerf.device{}, {}, {}".format(pose_world2nerf.device, global_frame_points.device, pose_world2nerf.device))
    point_at_nerf_frame = pose_world2nerf[:3,:3].double() @ global_frame_points.T.double() + pose_world2nerf[:3, 3][:, None].double()
    return point_at_nerf_frame.T.float()

def get_lidar_in_image_fov(pc_velo, pose, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=1.0, max_depth=100.0, mask=None):

    ''' Filter lidar points, keep those in image FOV '''
#    calib = torch.from_numpy(calib).cuda()
    calib[2][2]=1
    points = trans_world2nerf(pc_velo, pose) #right, up, back
    camera_points = torch.cat([points[..., 0:1], points[..., 1:2], points[..., 2:3]], -1)
    #camera_points = torch.cat([points[..., 0:1], -points[..., 1:2], -points[..., 2:3]], -1)
    pts_2d = torch.einsum('ij,nj->ni',calib, camera_points)
    pts_2d[...,0] = pts_2d[...,0]/(pts_2d[...,-1] + 1e-10)
    pts_2d[...,1] = pts_2d[...,1]/(pts_2d[...,-1] + 1e-10)

    if mask is not None:
        src = torch.from_numpy(mask.reshape(-1)).cuda()
        #idx = pts_2d[:,1].long() * 96 + pts_2d[:,0].long()
        #idx = torch.clamp(idx, 0, 6143)
        idx = pts_2d[:,1].long() * 640 + pts_2d[:,0].long()
        idx = torch.clamp(idx, 0, 640*480-1)
        mask_flag = src[idx]

        fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
            (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin) & (mask_flag==1)
    else:
        fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
            (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds_woDepthLimit = fov_inds & (camera_points[:,-1]>clip_distance)
    if max_depth>0:
        fov_inds = fov_inds_woDepthLimit & (camera_points[:,-1]<max_depth)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, points[fov_inds,:], fov_inds, pts_2d[fov_inds_woDepthLimit,:]
        #return imgfov_pc_velo, points[fov_inds,:], fov_inds, pts_2d[fov_inds,:]
    else:
        return imgfov_pc_velo

def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
            tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)

def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img

def dataset_viz():
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx)[:,0:3]
        calib = dataset.get_calibration(data_idx)

        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib, False)
        raw_input()
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        raw_input()

import matplotlib.pyplot as plt

def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(camera_image)
  #plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

def plot_points_on_image(frame_index, projected_points, camera_image=None, rgba_func=None,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  projected_points = projected_points.cpu()
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    #colors.append(rgba_func(point[2]))
    colors.append((point[2] / 75.2 * 255))

  ax = plt.gca()
  ax.xaxis.set_ticks_position('top')
  ax.invert_yaxis()

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
  os.makedirs('waymo_vis_point_on_image_top_all', exist_ok=True)
  plt.savefig('./waymo_vis_point_on_image_top_all/'+str(frame_index)+'_point_check_image.png')


if __name__=='__main__':
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d
    dataset_viz()
