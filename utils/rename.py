import glob
import os, sys
filenames = glob.glob(os.path.join('/mnt/lustre/caiyingjie/data/scannet/scans/scene0101_00_with_insert/*.npz'))
for filename in filenames:
    folder, fname = os.path.split(filename)
    aaa = fname[:-4].split('_')[-1]
    new_dst = os.path.join(folder, "results_pointnerf_scene0101_00_insert_"+aaa+".npz")
    cmd = "mv "+filename+ " "+new_dst
    os.system(cmd)
    print (cmd)
