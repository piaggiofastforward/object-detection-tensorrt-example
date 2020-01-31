# /// Copyright (c) 2020 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

import decoder
import cv2
import sys
import os
import yaml
import csv

decoder_path = sys.argv[1]
log_filepath = sys.argv[2]
output_dir = sys.argv[3]

#decode the bag
decoder.decode(decoder_path, log_filepath)
filename = os.path.split(log_filepath)[1]
base_filename = os.path.splitext(filename)[0]
output_dir = output_dir + '/' + base_filename + '/'

#prep output folders
realsense_dir = output_dir + 'realsense/'
grayscale_folder = realsense_dir + 'grayscale/'
depth_folder = realsense_dir + 'depth/'
rgb_folder_left = output_dir + 'rgb/left/'
rgb_folder_center = output_dir + 'rgb/center/'
rgb_folder_right = output_dir + 'rgb/right/'

if not os.path.exists(grayscale_folder):
    os.makedirs(grayscale_folder)
if not os.path.exists(depth_folder):
    os.makedirs(depth_folder)
if not os.path.exists(rgb_folder_left):
    os.makedirs(rgb_folder_left)
if not os.path.exists(rgb_folder_center):
    os.makedirs(rgb_folder_center)
if not os.path.exists(rgb_folder_right):
    os.makedirs(rgb_folder_right)

#read intrinsics
intrinsics = decoder.read_intrinsics()
yaml_dict = {'camera_intrinsics': intrinsics}
with open(realsense_dir + 'intrinsics.yaml', 'w') as outfile:
    yaml.dump(yaml_dict, outfile, default_flow_style=False, sort_keys=True)


#read realsense depth and state
depth_generator = decoder.next_depth_frame()
i = 0
headers = []
while(True):
    header, frame = next(depth_generator)
    if frame is not None:
        header.insert(0, i)
        headers.append(header)
        padded_frame = str(i).zfill(6)
        cv2.imwrite(depth_folder + padded_frame + '.png', frame)
        i += 1
    else:
        break

fields = ['Frame', 'Timestamp', 'Exposure', 'Gain', 'IR_Emitter_State', 'IR_Power']
with open(realsense_dir + 'times.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(fields)
    wr.writerows(headers)

#read odometry
fields, odoms = decoder.read_odometry()
with open(output_dir + 'odometry.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(fields)
    wr.writerows(odoms)

#read grayscale frames
grayscale_generator = decoder.next_grayscale_frame()
i = 0
while(True):
    header, frame = next(grayscale_generator)
    if frame is not None:
        padded_frame = str(i).zfill(6)
        cv2.imwrite(grayscale_folder + padded_frame + '.png', frame)
        i += 1
    else:
        break

#read rgb frames
rgb_generator = decoder.next_rgb_frame()
i = 0
while(True):
    header, frames = next(rgb_generator)
    if frames[0] is not None:
        padded_frame = str(i).zfill(6)

        cv2.imwrite(rgb_folder_left + padded_frame + '.png', frames[0])
        cv2.imwrite(rgb_folder_center + padded_frame + '.png', frames[1])
        cv2.imwrite(rgb_folder_right + padded_frame + '.png', frames[2])
        i += 1
    else:
        break


# #read grayscale frames
# grayscale_frames = decoder.read_realsense_grayscale()
# for i in range(len(grayscale_frames)):
#     padded_frame = str(i).zfill(6)
#     grayscale_frame = grayscale_frames[i]
#     cv2.imwrite(grayscale_folder + padded_frame + '.png', grayscale_frame)
#
# #read rgb frames
# rgb_frames = decoder.read_rgb()
# for i in range(len(rgb_frames)):
#     padded_frame = str(i).zfill(6)
#     camera_images = rgb_frames[i]
#     for j, image in enumerate(camera_images):
#         cv2.imwrite(rgb_folder + padded_frame + '-' + str(j) + '.png', image)