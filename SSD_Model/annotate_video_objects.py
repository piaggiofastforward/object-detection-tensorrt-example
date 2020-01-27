# DESCRIPTION
#
# Copyright (c) 2019 Piaggio Fast Forward (PFF), Inc.
# All Rights Reserved. Reproduction, publication,
# or re-transmittal not allowed without written permission of PFF, Inc.

import os
import ctypes
import time
import sys
import argparse
import json

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt

import utils.inference as inference_utils  # TRT/TF inference wrappers
import utils.model as model_utils  # UFF conversion
import utils.boxes as boxes_utils  # Drawing bounding boxes
import utils.coco as coco_utils  # COCO dataset descriptors
from utils.paths import PATHS  # Path management

import pycuda.driver as cuda
import pycuda.autoinit

# COCO label list
COCO_LABELS = coco_utils.COCO_CLASSES_LIST

# Model used for inference
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Confidence threshold for drawing bounding box
VISUALIZATION_THRESHOLD = 0.5

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT,
    8: trt.DataType.INT8
}

# Layout of TensorRT network output metadata
TRT_PREDICTION_LAYOUT = {
    "image_id": 0,
    "label": 1,
    "confidence": 2,
    "xmin": 3,
    "ymin": 4,
    "xmax": 5,
    "ymax": 6
}


def fetch_prediction_field(field_name, detection_out, pred_start_idx):
    """Fetches prediction field from prediction byte array.

    After TensorRT inference, prediction data is saved in
    byte array and returned by object detection network.
    This byte array contains several pieces of data about
    prediction - we call one such piece a prediction field.
    The prediction fields layout is described in TRT_PREDICTION_LAYOUT.

    This function, given prediction byte array returned by network,
    staring index of given prediction and field name of interest,
    returns prediction field data corresponding to given arguments.

    Args:
        field_name (str): field of interest, one of keys of TRT_PREDICTION_LAYOUT
        detection_out (array): object detection network output
        pred_start_idx (int): start index of prediction of interest in detection_out

    Returns:
        Prediction field corresponding to given data.
    """
    return detection_out[pred_start_idx + TRT_PREDICTION_LAYOUT[field_name]]


def analyze_prediction(detection_out, pred_start_idx, img_pil, currentFrame, filehandle, width, height, keep_list, filemodetxt):
    image_id = int(fetch_prediction_field(
        "image_id", detection_out, pred_start_idx))
    label = int(fetch_prediction_field("label", detection_out, pred_start_idx))
    confidence = fetch_prediction_field(
        "confidence", detection_out, pred_start_idx)
    xmin = fetch_prediction_field("xmin", detection_out, pred_start_idx)
    ymin = fetch_prediction_field("ymin", detection_out, pred_start_idx)
    xmax = fetch_prediction_field("xmax", detection_out, pred_start_idx)
    ymax = fetch_prediction_field("ymax", detection_out, pred_start_idx)
    if confidence > VISUALIZATION_THRESHOLD:
        class_name = COCO_LABELS[label]
        confidence_percentage = "{0:.0%}".format(confidence)
        print("Detected {} with confidence {}".format(
            class_name, confidence_percentage))
        boxes_utils.draw_bounding_boxes_on_image(
            img_pil, np.array([[ymin, xmin, ymax, xmax]]),
            display_str_list=["{}-{}: {}".format(
                class_name, pred_start_idx, confidence_percentage)],
            color=coco_utils.COCO_COLORS[label]
        )
        # update bonding box info to be saved
        found = False
        if(len(keep_list) != 0):
            for keep in keep_list:
                if(keep == class_name):
                    found = True
                if(keep == 'all'):
                    found = True
        else:
            found = True

        if(found == False):
            if(filemodetxt == True):
                framestring = "nan,nan,nan,nan\n"
            else:
                return
        if(filemodetxt == True):
            framestring = str(int(xmin*width)) + "," + \
                str(int(ymin*height)) + "," + \
                str(int(xmax*width)-int(xmin*width)) + "," + \
                str(int(ymax*height)-int(ymin*height)) + "\n"
        else:
            framestring = str(currentFrame) + "," + \
                class_name + "," + \
                confidence_percentage + "," + \
                str(pred_start_idx) + "," + \
                str(int(xmin*width)) + "," + \
                str(int(ymin*height)) + "," + \
                str(int(xmax*width)) + "," + \
                str(int(ymax*height)) + "\n"
        print(framestring)
        filehandle.write(framestring)


def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(
        description='Run object detection inference on input image.')
    parser.add_argument('--input_img_path', metavar='INPUT_IMG_PATH',
                        help='an image file to run inference on')
    parser.add_argument('-p', '--precision', type=int, choices=[32, 16, 8], default=32,
                        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1,
                        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--workspace_dir',
                        help='sample workspace directory')
    parser.add_argument('-fc', '--flatten_concat',
                        help='path of built FlattenConcat plugin')
    parser.add_argument('-d', '--calib_dataset', default='../VOCdevkit/VOC2007/JPEGImages',
                        help='path to the calibration dataset')
    parser.add_argument('-c', '--camera', default=True,
                        help='if True, will run webcam application')
    parser.add_argument('-k', '--keep', default='all',
                        help='Keep only classnames class1,class2,class3 ')
    parser.add_argument('-v', '--video_input',
                        help='feeds a video instead of the output of the webcam')
    parser.add_argument('-fmt', '--filemodetxt',
                        help='switch mode to txt mode of xmin,ymin,xwidth,ywidth')

    # Parse arguments passed
    args = parser.parse_args()

    # Set FlattenConcat TRT plugin path and
    # workspace dir path if passed by user
    if args.flatten_concat:
        PATHS.set_flatten_concat_plugin_path(args.flatten_concat)
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)

#    if args.video_input:
#        video_filepath = os.path.splitext(args.video_input)[0]

    try:
        os.makedirs(PATHS.get_workspace_dir_path())
    except:
        pass

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths()

    # Fetch TensorRT engine path and datatype
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.trt_engine_path = PATHS.get_engine_path(args.trt_engine_datatype,
                                                 args.max_batch_size)
    try:
        os.makedirs(os.path.dirname(args.trt_engine_path))
    except:
        pass

    return args


def main():

    # Parse command line arguments
    args = parse_commandline_arguments()
    keep_list = ['person']
    filter_list = []

    # Fetch .uff model path, convert from .pb
    # if needed, using prepare_ssd_model
    ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
    if not os.path.exists(ssd_model_uff_path):
        model_utils.prepare_ssd_model(MODEL_NAME)

    # Set up all TensorRT data structures needed for inference
    trt_inference_wrapper = inference_utils.TRTInference(
        args.trt_engine_path, ssd_model_uff_path,
        trt_engine_datatype=args.trt_engine_datatype,
        calib_dataset=args.calib_dataset,
        batch_size=args.max_batch_size)

    print("TRT ENGINE PATH", args.trt_engine_path)
    video_filepath = './video_capture'

    # Define the video stream
    if len(args.video_input) > 0:
        #       cap = cv2.VideoCapture('./2019-08-16_19-48-01_rgb.avi')
        print('Running videofile : ', args.video_input)
        cap = cv2.VideoCapture(args.video_input)
        video_filepath = os.path.splitext(args.video_input)[0]
    elif args.camera == True:
        # Change only if you have more than one webcams
        print('Running webcam:')
        cap = cv2.VideoCapture(0)
        # Change only if you have more than one webcams

    if args.filemodetxt:
        filemodetxt = True
    else:
        filemodetxt = False

    if args.keep:
        keep_list = args.keep.split(",")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    try:
        if not os.path.exists(video_filepath):
            os.makedirs(video_filepath)
    except OSError:
        exitError("failed to create output directory " + video_filepath)

    currentFrame = 0
    if filemodetxt == False:
        bb_filename = video_filepath + "/bounding_boxes.csv"
    else:
        bb_filename = video_filepath + "/bounding_boxes.txt"
    bb_file_handle = open(bb_filename, "w")
    # write header
    if filemodetxt == False:
        bb_file_handle.write(
            "frame_number, class_name, confidence, idx, xmin, ymin, xmax, ymax\n")
    # Loop for running inference on frames from the webcam
    while True:
        # Read frame from camera (and expand its dimensions to fit)
        ret, image_np = cap.read()
        if not ret:
            break

        # Actually run inference
        detection_out, keep_count_out = trt_inference_wrapper.infer_webcam(
            image_np)

        name = video_filepath + "/frame" + str(currentFrame) + ".jpg"
        # create and open a file where each frame bounding boxes will be stored

        # Overlay the bounding boxes on the image
        # let analyze_prediction() draw them based on model output
        img_pil = Image.fromarray(image_np)
        prediction_fields = len(TRT_PREDICTION_LAYOUT)

        for det in range(int(keep_count_out[0])):
            analyze_prediction(detection_out, det *
                               prediction_fields, img_pil, currentFrame, bb_file_handle, width, height, keep_list, filemodetxt)

        final_img = np.asarray(img_pil)

        # Display output
        cv2.imshow('object detection', final_img)

        print("Creating..." + name)
        cv2.imwrite(name, final_img)
        currentFrame += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    bb_file_handle.flush()
    bb_file_handle.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
