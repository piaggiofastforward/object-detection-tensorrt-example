# DESCRIPTION
# Sync bag and automated annotation results
# Copyright (c) 2019 Piaggio Fast Forward (PFF), Inc.
# All Rights Reserved. Reproduction, publication,
# or re-transmittal not allowed without written permission of PFF, Inc.

import os
import argparse
import cv2
import csv
import numpy
import decoder as dc

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(
        description='Run object detection inference on input image.')
    parser.add_argument('-d', '--calib_dataset', default='../VOCdevkit/VOC2007/JPEGImages',
                        help='path to the calibration dataset')
    parser.add_argument('-g', '--grayscale', action='store_true',
                        help='use grayscale instead of RGB data')
    parser.add_argument('-bd','--bags_directory', default='/media/bags', 
                        help='location of bags for annotation')
    parser.add_argument('-aad','--auto_annotation_directory', default='/media/auto_annotated_results', 
                        help='location of results folder')
    parser.add_argument('-rd','--results_directory', default='/media/manual_annotated_results', 
                        help='location of results folder')

    # Parse arguments passed
    args = parser.parse_args()

    return args


def main():

    args = parse_commandline_arguments()

    #get list of bags
    bags_path = dc.get_bags_in_folder(args.bags_directory)
    decoder_path = 'logdecoder'

    for bag_path in bags_path:
        print(bag_path)

        #decode the current bag
        dc.decode(decoder_path,bag_path)

        #use the current bag name to generate the auto annotation results file name
        bag_name = os.path.split(bag_path)[1]
        if args.grayscale:
            frames = dc.read_realsense_grayscale()
            auto_annotation_file = os.path.join(args.auto_annotation_directory, 'AAR_'+bag_name[:len(bag_name)-4]+'_gray.csv')
        else:
            frames = dc.read_rgb()
            auto_annotation_file = os.path.join(args.auto_annotation_directory,'AAR_'+bag_name[:len(bag_name)-4]+'.csv')

        #confirm the auto annotation file exists
        if os.path.exists(auto_annotation_file):
            print(auto_annotation_file)
            auto_annotation_file_handler = open(auto_annotation_file,"r")
        else:
            print('cannot find auto annotation file')
            break

        #try making the the output directory
        try:
            if not os.path.exists(args.results_directory):
                os.makedirs(args.results_directory)
        except OSError:
            exitError("failed to create output directory " + args.results_directory)

        header = auto_annotation_file_handler.readline()
        line = []

        for i in range(len(frames)):
            image = frames[i]

            #print lines in auto annotation results that correspond to the current frame
            if not line:
                line = auto_annotation_file_handler.readline()
            while True:
                linelist = line.split(",")
                if linelist[0]:
                    if int(linelist[0]) == i:
                        print(line)
                        line = auto_annotation_file_handler.readline()
                    else:
                        break
                else:
                    break

            #show image frame                        
            cv2.imshow('images',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        #close the current file so we can open the next one
        auto_annotation_file_handler.flush()
        auto_annotation_file_handler.close()


if __name__ == '__main__':
    main()