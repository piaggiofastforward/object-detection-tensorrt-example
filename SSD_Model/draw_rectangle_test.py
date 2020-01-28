# /// Copyright (c) 2020 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

from __future__ import division
from __future__ import print_function

# import the necessary packages
import cv2
import argparse

# Initialize the list of reference points
ref_point = []

started_drawing_triangle = False
image = None
clone = None

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, started_drawing_triangle, image, clone

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        started_drawing_triangle = True
        ref_point = [(x, y)]

    if started_drawing_triangle and event == cv2.EVENT_MOUSEMOVE:
        new_last_point  = (x, y)
        print("Down and moving at " + "(%d, %d)" % new_last_point)

        # draw a rectangle around the region of interest
        image = clone.copy()
        cv2.rectangle(image, ref_point[0], new_last_point, (0, 255, 0), 2)
        cv2.imshow("image", image)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        started_drawing_triangle = False
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

    # Paths correctness verifier

def resolve_path(input_path):
    from os import path, getcwd
    if input_path is None or len(input_path) == 0:
        input_path = getcwd()
    path_of_interest = path.dirname(input_path)  # Do this in case a filename was passed instead
    resolved_user_expanded_path = path.expanduser(path_of_interest)
    resolved_realpath = path.realpath(resolved_user_expanded_path)
    return resolved_realpath

def verify_path(input_path):
    """
    Returns a validated path for the input_path if no errors occur
    """
    import sys
    from os import path
    error = False
    
    resolved_realpath = resolve_path(input_path)
    if not path.exists(resolved_realpath):
        error = True

    if error:
        print("An error occured when verifying the path %s" % (input_path))
        sys.exit(1)
    else:
        return resolved_realpath

            
def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(
        description='Run object detection inference on input image.')
    
    parser.add_argument('-i', '--images_input_path', help='Path to folder containing images', type=str)
    parser.add_argument('-v', '--video_input', help='Path to a video file instead of images sequence', type=str)
    parser.add_argument('-l', '--labels', help='The labels filename containing the bounding box information as well as frame number', type=str, required=True)
    parser.add_argument('-o', '--output_folder', help='Path to output folder', type=str)

    # Parse arguments passed
    args = parser.parse_args()
    return args


def make_sure_path_exists(input_path):
    from os import makedirs
    import errno
    try:
        resolved_realpath = resolve_path(input_path)
        makedirs(resolved_realpath)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    return resolved_realpath

def splitfn(fn):
    from os.path import split, splitext
    path_root, fn_only = split(fn)
    name, ext = splitext(fn_only)
    return path_root, name, ext

def get_images(filename_template, indices_list = [], show_images = False, return_names_only = False):
    '''
    @param indices_list: Returns only those images indices from the entire list. If this list is empty (default), all images read are returned
    @note: all images files acquired by glob will be read and shown (however), but only those indexed in the list (if any) will be returned
    @return: A list of the retrieved images (based on an index list, if any) from the filenames template. If the return_names_only is set to True, only the names of the images will be retrieved
    '''
    #===========================================================================
    # from glob import glob
    # img_names = glob(filename_template)
    #===========================================================================
    # It's faster to retrieve files from a directory with "fnmatch":
    import fnmatch
    from os import listdir
    from os.path import split, join
    import warnings

    path_to_files, pattern_filename = split(filename_template)
    img_names = fnmatch.filter(listdir(path_to_files), pattern_filename)

    if indices_list is None or len(indices_list) == 0:
        l = len(img_names)
        indices_list = range(l)
    else:
        l = len(indices_list)

    img_names_list_all = [join(path_to_files, img_name) for img_name in img_names]
    img_names_list = l * [None]
    for i, img_index in enumerate(indices_list):
        img_names_list[i] = img_names_list_all[img_index]

    if return_names_only:
        return img_names_list

    images = l * [None]
    for i, fn in enumerate(img_names_list):
        try:
            # fn = img_names[i] # when using glob
            # fn = join(path_to_files, img_names[i])  # When using fnmatch
            print('Reading %s...' % fn, end = "")
            img = cv2.imread(fn)
            if img is not None:
                print("success")
                images[i] = img
                if show_images:
                    path, name, ext = splitfn(fn)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, img)
            else:
                print("failed!")
        except:
            warnings.warn("Warning...image index %d not found at %s" % (i, __name__))

    if show_images:
        cv2.waitKey(1)

    # We want the whole list!, so this is not good any more
#     if len(indices_list) == 0:
#         return images  # all
#     else:
#         return list(np.take(images, indices_list, 0))

    return images  # all even if None

def main():
    global image, clone

    import pandas as pd 
    import sys
    from os import path
    args = parse_commandline_arguments()

    if not path.isfile(args.labels):
        print("Input bounding boxes (labels) file is missing!")
        sys.exit(1)
    input_bounding_boxes_file = args.labels
        
    output_folder = make_sure_path_exists(args.output_folder)
    output_gt_filename = path.join(output_folder, "gt.csv") 
    
    use_video_file = True
    if args.video_input is None:
        if not (args.images_input_path is None):
            use_video_file = False
        else:
            print("Path to video nor image sequence does not exist")
            sys.exit(1)
    
    images_list = []  # FIXME: doing this is very memory intensive because we are reading all the images at once
    # However, we need to do this in order to allow for the "backward/forward" functionality during annotation
    if use_video_file:
        cam = cv2.VideoCapture(args.video_input)
        video_frame_read_status = True
        while video_frame_read_status:
            video_frame_read_status, img = cam.read()
            if video_frame_read_status:
                images_list.append(img)
    else:
        # resolve images in path
        input_img_folder = verify_path(args.images_input_path)
        input_imgs_template = path.join(input_img_folder, "*")
        images_list = get_images(filename_template=input_imgs_template, indices_list=[], show_images=False, return_names_only=False)
    
    truth_content = pd.read_csv(input_bounding_boxes_file) 
    last_frame_number = truth_content.frame_number[len(truth_content.frame_number) - 1]
    
    # CHECKME: why is this happening whith this particular data set?
    working_num_of_frames = last_frame_number + 1  # Account for zero, too
    if working_num_of_frames != len(images_list):
        print("Read number of frames does NOT match ground truth information")
        working_num_of_frames = min([working_num_of_frames, len(images_list)])
        print("Using only %d frames" % (working_num_of_frames))
        sys.exit(1)  # CHECME: exiting shouldn't be necessary

    # TODO: Just reading the first image
    image = images_list[0]
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)
    
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
    
        # press 'r' to reset the window
        if key == ord("r"):
            image = clone.copy()
    
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
    # close all open windows
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()

