# /// Copyright (c) 2019 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

import subprocess
import os
import numpy as np
import cv2
import struct
import csv
import shutil


# def read_depth(filename='/tmp/decoding/log-depth.dat'):
#     all_frames = []
#     timestamps = []
#     data = np.fromfile(filename, dtype='uint8')
#     frame_size = 480 * 640 * 2 + 5 * 8 #5 fields of 8 bytes each precede the depth data
#     frame_count = data.shape[0] // (frame_size)
#     frames = data.reshape(frame_count, frame_size)
#
#     for i in range(frame_count):
#         frame = frames[i]
#         # timestamps.append(frame[:8].view(np.int64))
#         frame = frame[8*5 :]
#         frame = np.array(frame.view(np.uint16))
#         frame = frame.reshape(480, 640)
#         frame = frame.astype(np.uint16)
#         all_frames.append(frame)
#     return all_frames

def next_depth_frame(filename='/tmp/decoding/log-depth.dat'):
    # all_frames = []
    # timestamps = []
    #
    # f = open(filename, "rb")
    # f.seek(0, os.SEEK_SET)
    #
    # while(True):
    #
    #     # header = np.fromfile(filename, depth_header, count=1)
    #
    #     try:
    #         time = np.fromfile(f, np.uint8, count=1)
    #         exposure = np.fromfile(f, np.uint8, count=1)
    #         gain = np.fromfile(f, np.uint8, count=1)
    #         ir_state = np.fromfile(f, np.uint8, count=1)
    #         ir_power = np.fromfile(f, np.uint8, count=1)
    #         frame = np.fromfile(f, np.uint16, 480 * 640)
    #         frame = frame.reshape(480, 640)
    #         yield frame
    #     except:
    #         break

    chunksize = 480 * 640 * 2 + 5 * 8
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                frame = np.frombuffer(chunk, dtype=np.uint8)
                timestamp = frame[:8*1]
                timestamp = np.array(timestamp.view(np.int64))[0]

                exposure = frame[8*1:8*2]
                exposure = np.array(exposure.view(np.int64))[0]

                gain = frame[8*2:8*3]
                gain = np.array(gain.view(np.int64))[0]

                ir_state = frame[8*3:8*4]
                ir_state = np.array(ir_state.view(np.int64))[0]

                ir_power = frame[8*4:8*5]
                ir_power = np.array(ir_power.view(np.int64))[0]

                frame = frame[8 * 5:]
                frame = np.array(frame.view(np.uint16))
                frame = frame.reshape(480, 640)
                frame = frame.astype(np.uint16)

                header = [timestamp, exposure, gain, ir_state, ir_power]
                yield header, frame
            else:
                break

    yield None, None

# def read_realsense_grayscale(filename='/tmp/decoding/log-grayscale.dat'):
#     all_frames = []
#     data = np.fromfile(filename, dtype='uint8')
#     frame_size = 480 * 640 + 8
#     frame_count = data.shape[0] // (frame_size)
#
#     frames = data.reshape(frame_count, frame_size)
#
#     for i in range(frame_count):
#         frame = frames[i]
#         frame = frame[:-8]
#         frame = np.array(frame.view(np.uint8))
#         frame = frame.reshape(480, 640)
#         frame = frame.astype(np.uint8)
#         all_frames.append(frame)
#     return all_frames

def next_grayscale_frame(filename='/tmp/decoding/log-grayscale.dat'):
    chunksize = 480 * 640 + 8
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                frame = np.frombuffer(chunk, dtype=np.uint8)

                timestamp = frame[-8:]
                timestamp = np.array(timestamp.view(np.int64))

                frame = frame[:-8]
                frame = np.array(frame.view(np.uint8))
                frame = frame.reshape(480, 640)
                frame = frame.astype(np.uint8)

                yield timestamp, frame
            else:
                break

    yield None, None

# def read_rgb(filename='/tmp/decoding/log-rgb.dat'):
#     all_frames = []
#     data = np.fromfile(filename, dtype='uint8')
#     frame_size = 480 * 1920 + 8
#     frame_count = data.shape[0] // (frame_size)
#
#     frames = data.reshape(frame_count, frame_size)
#
#     for i in range(frame_count):
#         frame = frames[i]
#         frame = frame[:-8]
#         frame = np.array(frame.view(np.uint8))
#         frame = frame.reshape(480, 1920)
#         frame = frame.astype(np.uint8)
#         frame1 = frame[:,:640]
#         frame2 = frame[:,640:1280]
#         frame3 = frame[:,1280:]
#
#         frame1 = cv2.cvtColor(frame1, cv2.COLOR_BAYER_BG2RGB)
#         frame2 = cv2.cvtColor(frame2, cv2.COLOR_BAYER_BG2RGB)
#         frame3 = cv2.cvtColor(frame3, cv2.COLOR_BAYER_BG2RGB)
#         all_frames.append([frame1, frame2, frame3])
#     return all_frames

def next_rgb_frame(filename='/tmp/decoding/log-rgb.dat'):
    chunksize = 480 * 1920 + 8
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                frame = np.frombuffer(chunk, dtype=np.uint8)
                timestamp = frame[-8:]
                timestamp = np.array(timestamp.view(np.int64))
                frame = frame[:-8]
                frame = np.array(frame.view(np.uint8))
                frame = frame.reshape(480, 1920)
                frame = frame.astype(np.uint8)

                frame1 = frame[:, :640]
                frame2 = frame[:, 640:1280]
                frame3 = frame[:, 1280:]

                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BAYER_BG2RGB)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BAYER_BG2RGB)
                frame3 = cv2.cvtColor(frame3, cv2.COLOR_BAYER_BG2RGB)
                yield timestamp, [frame1, frame2, frame3]
            else:
                break

    yield None, [None, None, None]

def read_intrinsics(filename='/tmp/decoding/log-intrinsics.dat'):
    intrinsics = {}
    with open(filename, 'rb') as f:
        bytes = f.read()
        if bytes != b'':
            intrinsics_fields = ['timestamp', 'w', 'h', 'ppx', 'ppy', 'fx', 'fy', 'model', 'coeffs1', 'coeffs2', 'coeffs3', 'coeffs4', 'coeffs5']
            types = ['Q', 'Q', 'Q', 'd', 'd', 'd', 'd', 'Q', 'd', 'd', 'd', 'd', 'd']
            seek = 0
            for i, field in enumerate(intrinsics_fields):
                intrinsics[field] = struct.unpack(types[i], bytes[seek:seek+8])[0]
                seek += 8
            intrinsics.pop('timestamp')
    return intrinsics

# def read_realsense_state(filename='/tmp/decoding/log-depth.dat'):
#     # all_states = []
#
#     fields = ['Timestamp', 'Exposure', 'Gain', 'IR_Emitter_State', 'IR_Power']
#     types = ['Q', 'Q', 'Q', 'Q', 'Q']
#     with open(filename, 'rb') as f:
#         bytes = f.read()
#
#         seek = 0
#         count = 0
#         while(True):
#             try:
#                 states = [count]
#                 for i, field in enumerate(fields):
#                     states.append(struct.unpack(types[i], bytes[seek:seek+8])[0])
#                     seek += 8
#                 seek += 640*480*2
#                 count += 1
#                 yield states
#             except:
#                 break
#
#     # fields.insert(0, 'Frame')
#     # return fields, all_states
#     yield None

def read_odometry(filename='/tmp/decoding/log-odom.dat'):
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

def decode(decoder_path, bag_path, decoded_bag_path='/tmp/decoding/'):
    if os.path.exists(decoded_bag_path):
        shutil.rmtree(decoded_bag_path)
    if not os.path.exists(decoded_bag_path):
        os.makedirs(decoded_bag_path)
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

    decode(decoder_path, '../../../logs/bag.klv')

    frame_generator = next_rgb_frame()

    while(True):
        timestamp, frames = next(frame_generator)
        if frames[0] is not None:
            cv2.imshow('frame', frames[1])
            cv2.waitKey(-1)
        else:
            break

if __name__ == "__main__":
    main()
