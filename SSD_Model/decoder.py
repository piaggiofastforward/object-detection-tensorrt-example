# /// Copyright (c) 2019 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

import subprocess
import os
import numpy as np
import cv2
import struct
import csv

def read_depth(filename='/tmp/log-depth.dat'):
    all_frames = []
    timestamps = []
    data = np.fromfile(filename, dtype='uint8')
    frame_size = 480 * 640 * 2 + 5 * 8 #7 fields of 8 bytes each precede the depth data
    frame_count = data.shape[0] // (frame_size)

    frames = data.reshape(frame_count, frame_size)

    for i in range(frame_count):
        frame = frames[i]
        # timestamps.append(frame[:8].view(np.int64))
        frame = frame[8*5:]
        frame = np.array(frame.view(np.uint16))
        frame = frame.reshape(480, 640)
        frame = frame.astype(np.uint16)
        all_frames.append(frame)
    return all_frames

def read_realsense_grayscale(filename='/tmp/log-images.dat'):
    all_frames = []
    data = np.fromfile(filename, dtype='uint8')
    frame_size = 480 * 640 + 8
    frame_count = data.shape[0] // (frame_size)

    frames = data.reshape(frame_count, frame_size)

    for i in range(frame_count):
        frame = frames[i]
        frame = frame[8:]
        frame = np.array(frame.view(np.uint8))
        frame = frame.reshape(480, 640)
        frame = frame.astype(np.uint8)
        all_frames.append(frame)
    return all_frames

def read_intrinsics(filename='/tmp/log-intrinsics.dat'):
    with open(filename, 'rb') as f:
        bytes = f.read()
        intrinsics_fields = ['timestamp', 'w', 'h', 'cx', 'cy', 'fx', 'fy', 'model', 'coeffs1', 'coeffs2', 'coeffs3', 'coeffs4', 'coeffs5']
        types = ['Q', 'Q', 'Q', 'd', 'd', 'd', 'd', 'Q', 'd', 'd', 'd', 'd', 'd']
        intrinsics = {}
        seek = 0
        for i, field in enumerate(intrinsics_fields):
            intrinsics[field] = struct.unpack(types[i], bytes[seek:seek+8])[0]
            seek += 8
        intrinsics['d_scale'] = 0.001

    return intrinsics

def read_realsense_state(filename='/tmp/log-depth.dat'):
    all_states = []

    fields = ['Timestamp', 'Exposure', 'Gain', 'IR_Emitter_State', 'IR_Power']
    types = ['Q', 'Q', 'Q', 'Q', 'Q']
    with open(filename, 'rb') as f:
        bytes = f.read()

        seek = 0
        count = 0
        while(True):
            try:
                states = [count]
                for i, field in enumerate(fields):
                    states.append(struct.unpack(types[i], bytes[seek:seek+8])[0])
                    seek += 8
                seek += 640*480*2
                all_states.append(states)
                count += 1
            except:
                break

    fields.insert(0, 'Frame')
    return fields, all_states

def read_odometry(filename='/tmp/log-odom.dat'):
    all_odom = []

    fields = ['Timestamp', 'X', 'Y', 'V_Linear', 'V_Angular', 'Shift', 'Roll', 'Pitch', 'Yaw']
    types = ['Q', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd']
    with open(filename, 'rb') as f:
        bytes = f.read()

        seek = 0
        count = 0
        while(True):
            try:
                states = []
                for i, field in enumerate(fields):
                    states.append(struct.unpack(types[i], bytes[seek:seek+8])[0])
                    seek += 8
                all_odom.append(states)
                count += 1
            except:
                break

    return fields, all_odom

def decode(decoder_path, bag_path, decoded_bag_path='/tmp/'):
    # path, file = os.path.split(log)
    command = decoder_path + ' ' + bag_path + ' ' + decoded_bag_path
    output = subprocess.call(command, shell=True)

def get_bags_in_folder(dir):
    fullpaths = []
    for path in os.listdir(dir):
        fullpaths.append(os.path.join(dir, path))
    return fullpaths

def write_results(results, filename):
    #results is a list of lists where each result is [x,y,w,h]
    with open(filename, "w") as f:
        wr = csv.writer(f)
        wr.writerows(results)


def main():
    #this shows some example usage
    decoder_path = '../../bin/logdecoder'

    decode(decoder_path, '/home/derekmccoll/data/bags/log-2020-01-21-21-32-22.klv')
    intrinsics = read_intrinsics()
    fields, states = read_realsense_state()
    fields, odoms = read_odometry()

    #decode(decoder_path, '../../../logs/depth_and_grayscale/logs/log-2019-11-8-21-0-46')
    depth_frames = read_depth()
    grayscale_frames = read_realsense_grayscale()

    for i in range(len(depth_frames)):
        cv2.imshow('depth', depth_frames[i])
        cv2.imshow('grayscale', grayscale_frames[i])
        cv2.waitKey(-1)


if __name__ == "__main__":
    main()
