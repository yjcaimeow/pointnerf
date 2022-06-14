import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import imageio
import torch
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
from termcolor import colored, cprint
tf.enable_eager_execution()
import cv2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('../')
import models.mvs.mvs_utils as mvs_utils
#from .data_utils import get_dtu_raydir
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
  #plt.imshow(camera_image)
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")

def plot_points_on_image(frame_index, projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  #plot_image(camera_image)

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

def draw(poses):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    ax, ay, az, bx, by, bz = [],[],[],[],[],[]
    for pose in poses:
        t = pose[0:3, -1].reshape(3,1)
        R = pose[0:3, 0:3]
        a_tra = t
        b_tra = np.dot(-R.T, t)
        ax.append(a_tra[0][0])
        ay.append(a_tra[1][0])
        az.append(a_tra[2][0])

        bx.append(b_tra[0][0])
        by.append(b_tra[1][0])
        bz.append(b_tra[2][0])

    ax = np.array(ax)
    ay = np.array(ay)
    az = np.array(az)
    bx = np.array(bx)
    by = np.array(by)
    bz = np.array(bz)

    fig = plt.figure()
    ax_fig = Axes3D(fig)
    ax_fig.scatter(ax, ay, az)
    ax_fig.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax_fig.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax_fig.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.savefig("filename_a_hhh.png")

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(bx, by, bz)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.savefig("filename_b_hhh.png")

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses, c2w

pose_nerf2camera = np.array([
        [0, 0,-1, 0],
        [-1,0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1.],
    ])
pose_camera2nerf = np.linalg.inv(pose_nerf2camera).astype(np.float32)

def load_waymo_data(FILENAME, frames_length=200,  half_res=True, step=10, \
                    load_point=True, start_frame=0, \
                    split='test', scale_factor=10, args=None, device = None, vox_res=100, width=1920, height=1280):
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    all_imgs, all_poses, all_depths, all_mask, all_points = [], [], [], [], []
    camposes=[]
    centerdirs=[]
    img_wh = (int(width//scale_factor), int(height//scale_factor))
    centerpixel=np.asarray(img_wh).astype(np.float32)[None,:] // 2
    frame_index = 0
    all_id_list = []
    for index, data in enumerate(dataset):
        all_id_list.append(index)
        if index<start_frame:
            continue
        if frame_index>=frames_length and frames_length!=-1:
            break
        frame_index = frame_index+1

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        ''' image load '''
        front_camera = frame.images[0]
        data = frame.context
        #pose_camera2world    = np.reshape(np.array(frame.images[0].pose.transform, np.float32), (4, 4))
        pose_vehicle2world   = np.reshape(np.array(frame.pose.transform, np.float32), (4, 4))
        img = (np.array(tf.image.decode_jpeg(front_camera.image)) / 255.).astype(np.float32)
        #cv2.imwrite(str(index)+'_front_img.png', img*255)

        if index == 0  or frame_index==1:
            intrinsic = data.camera_calibrations[0].intrinsic
            pose_camera2vehicle= np.array(data.camera_calibrations[0].extrinsic.transform,dtype=np.float32).reshape(4,4) #camera-vehicle from the sensor frame to the vehicle frame.
            pose_vehicle2camera = np.linalg.inv(pose_camera2vehicle).astype(np.float32)
            focal = intrinsic[0]
            K = np.array([ \
                          [intrinsic[0], 0, intrinsic[2]], \
                          [0, intrinsic[0], intrinsic[3]], \
                          [0, 0, 1]], dtype=np.float32)
            W, H = data.camera_calibrations[0].width, data.camera_calibrations[0].height

        '''img undist '''
        undist_img = cv2.undistort(img, K, np.asarray(intrinsic[4:9]), None, K)

        ''' lidar point load '''
        # if load_point and args.render_only==False:
        if load_point and index%10!=0:
            print ('------- load_point data -------', index)
            (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,
                range_images,
                camera_projections,
                range_image_top_pose)
            points_all = np.concatenate(points, axis=0).astype(np.float32)
            cp_points_all = np.concatenate(cp_points, axis=0).astype(np.float32)
            images = sorted(frame.images, key=lambda i:i.name)
            cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1).astype(np.float32)
            # cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

            #points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True) # !!! sqrt(x,y,z)
            points_all_tensor = tf.constant(points_all, dtype=tf.float32)
            cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

            mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

            cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

            point_at_vehicle_frame = np.asarray(points_all_tensor,dtype=np.float32)

            point_at_world_frame = pose_vehicle2world[:3,:3] @ point_at_vehicle_frame.T + pose_vehicle2world[:3, 3][:, None]
            point_at_world_frame = point_at_world_frame.T

            if vox_res > 0:
                # print("world_xyz", point_at_world_frame.shape, torch.min(point_at_world_frame.view(-1,3), dim=-2)[0], torch.max(point_at_world_frame.view(-1,3), dim=-2)[0])
                # np.savetxt('original.txt', point_at_world_frame.numpy())
                point_at_world_frame = mvs_utils.construct_vox_points_xyz(torch.from_numpy(point_at_world_frame), vox_res)
                #print("world_xyz", point_at_world_frame.shape, torch.min(point_at_world_frame.view(-1,3), dim=-2)[0], torch.max(point_at_world_frame.view(-1,3), dim=-2)[0])
                # np.savetxt('after.txt', point_at_world_frame.numpy())
                # exit()
            all_points.append(point_at_world_frame)
        pose_camera2world = pose_vehicle2world @ pose_camera2vehicle
        c2w = pose_camera2world
        campos = c2w[:3, 3]
        camrot = c2w[:3,:3]
        #raydir = get_dtu_raydir(centerpixel, K, camrot, True)
        camposes.append(campos)
        #centerdirs.append(raydir)

        all_imgs.append(undist_img)
        all_poses.append(pose_camera2world)
    if load_point :
        all_points = np.concatenate(all_points, axis=0)
        #if args.sample_type=='random':
        #    print ('sample_type is random success!')
        #    sample_indexs = np.random.choice(np.arange(len(all_points)), size=args.point_number, replace=False)
        #    all_points = all_points[sample_indexs]
        #elif args.sample_type=='fps':
        #    print ('sample_type is fps success!')
        #    from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import FPS
        #    fps = FPS()
        #    #original_indexs = all_points[...,0:1]
        #    #all_points, all_indexs = fps(torch.tensor(all_points[None, ...][:,:,1:4]).contiguous(), args.point_number)
        #    all_points, _ = fps(torch.tensor(all_points[None, ...]).contiguous(), args.point_number)
        #    all_points = all_points.squeeze().cpu().numpy()
        #    #all_indexs = original_indexs[all_indexs.squeeze().cpu().numpy()]
        #    #all_points = np.concatenate((all_indexs, all_points), axis=-1).astype(np.float32)

    imgs = np.asarray(all_imgs, dtype=np.float32)
    poses = np.asarray(all_poses, dtype=np.float32)
    camposes=np.stack(camposes, axis=0)
    #centerdirs=np.concatenate(centerdirs, axis=0)
    poses = np.concatenate([-poses[:, :, 1:2], poses[:, :, 2:3], -poses[:, :, 0:1], poses[:, :, 3:4]], 2)

    test_id_list = all_id_list[::step]
    train_id_list = [all_id_list[i] for i in range(len(all_id_list)) if (i % step) !=0]
    if half_res:
        H = H//scale_factor
        W = W//scale_factor
        focal = focal/scale_factor
        K = K/scale_factor
        K[2][2] = 1
        ### for target img size
        img_H, img_W = H*4, W*4
        imgs_half_res = np.zeros((imgs.shape[0], img_H, img_W, 3))

        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img,  (img_W, img_H), interpolation=cv2.INTER_AREA)
        imgs = np.asarray(imgs_half_res, dtype=np.float32)
    return torch.from_numpy(imgs), torch.from_numpy(poses), [H, W, focal], torch.from_numpy(K), torch.from_numpy(all_points), all_id_list, test_id_list, train_id_list, \
            torch.from_numpy(camposes)
            #torch.from_numpy(camposes), torch.from_numpy(centerdirs)

def main(filename):
    img, poses, hwf, k, pcd, *_ = load_waymo_data(filename)
    dstroot = filename.replace('tfrecord', 'folder')
    os.makedirs(dstroot, exist_ok=True)
    np.save(os.path.join(dstroot, 'pcd.npy'), pcd)
    for index in range(len(img)):
        np.savez(os.path.join(dstroot, str(index).zfill(3)+'_info.npz'), image=img[index], pose=poses[index], hwf=hwf, k=k)

from multiprocessing import Pool
if __name__ == '__main__':
    import glob
    filenames = glob.glob('/mnt/lustre/caiyingjie/data/selected_waymo/*.tfrecord')
    with Pool(150) as p:
        print(p.map(main, filenames))
