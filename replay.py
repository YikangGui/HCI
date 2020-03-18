#!/bin/python
import os
import shutil

json_dir_name = 'data/all_data'  # input directory
video_dir_name = 'data/videos'  # output directory
tmp_img_dir_name = 'data/tmp_imgs/'  # DO NOT CHANGE

if not os.path.isdir(video_dir_name):
    os.mkdir(video_dir_name)

json_dir = os.listdir(json_dir_name)
json_dir.sort()
if os.path.isdir(tmp_img_dir_name):
    shutil.rmtree(tmp_img_dir_name)

for json_file in json_dir:
    if json_file[-5:] != '.json':
        continue
    print(json_file)
    print('please waiting, it keeps about 5min to generate one video... \n\n')

    # 1. Generate tmp PNGs
    os.mkdir(tmp_img_dir_name)
    os.system('./bin/replay %s %s' % (os.path.join(json_dir_name, json_file), tmp_img_dir_name))

    # 2. Generate MP4 from tmp PNGs
    os.system('ffmpeg -framerate 30 -i ' + tmp_img_dir_name + '%04d.png ' + os.path.join(video_dir_name,
                                                                                         json_file.replace('.json',
                                                                                                           '.mp4')))

    # 3. Remove tmp PNGs
    shutil.rmtree(tmp_img_dir_name)
    print('success!\n')