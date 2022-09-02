import sys
import glob
import os

folders = glob.glob('/mnt/cache/caiyingjie/data/scannet/scans/*')

for folder in folders:
    print (folder)
    _, name = os.path.split(folder)
    merge_cmd = "ffmpeg -f image2 -i "+folder+'/exported/color/%d.jpg /mnt/lustre/caiyingjie/'+name+'.mp4'
    os.system(merge_cmd)
    print(merge_cmd+' successed.')
