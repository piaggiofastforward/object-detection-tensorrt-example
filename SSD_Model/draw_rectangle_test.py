# /// Copyright (c) 2020 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

from __future__ import division
from __future__ import print_function

# import the necessary packages
import cv2
import numpy as np
import argparse

# Initialize the shared lists
ref_points_list = []  # reference points
current_gt_in_list = []  # current ground truth entries
box_colors_list = []

started_drawing_triangle = False
started_drawing_triangle_and_moused_moved = False
image = None
clone = None


def rgb2bgr_color(rgb_color):
    return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))


def draw_points(img_input, points_uv_coords, num_valid_points=None, color_rgb=None, thickness=1):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param points_uv_coords: FIXME: the uv coordinates list or ndarray must be of shape (n, 2) for n points.
    Note that the coordinates will be expressed as integers while visualizing
    @param color_rgb: a 3-tuple of the RGB color_rgb for these points
    '''
    if color_rgb == None:
        color_rgb = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color_rgb from RGB into BGR
        color_rgb = rgb2bgr_color(color_rgb)

    if num_valid_points == None:
        num_valid_points = len(points_uv_coords)

    for i in range(num_valid_points):
        pt = points_uv_coords[i]
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            print("nan cannot be drawn!")
        else:
            try:  # TODO: also, out of border points cannot be drawn!
                pt_as_tuple = (int(pt[0]), int(pt[1]))  # Recall: (pt[0],pt[1]) # (x, u or col and y, v or row)
                cv2.circle(img_input, pt_as_tuple, 2, color_rgb, thickness, 8, 0)
            except:
                print("Point", pt_as_tuple, "cannot be drawn!")


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_points_list, current_gt_in_list, started_drawing_triangle, started_drawing_triangle_and_moused_moved, image, clone

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        started_drawing_triangle = True
        started_drawing_triangle_and_moused_moved = False
        ref_points_list = [(x, y)]
        
        # draw the starting point
        image = clone.copy()
        draw_points(image, points_uv_coords=ref_points_list, num_valid_points=1, color_rgb=(255, 0, 0), thickness=2)

        # Check if point is within some existing box
        valid_boxes_indices = []
        for entry_idx, gt_entry in enumerate(current_gt_in_list):
            bb_x_min = int(gt_entry['xmin'])
            bb_y_min = int(gt_entry['ymin'])
            bb_x_max = int(gt_entry['xmax'])
            bb_y_max = int(gt_entry['ymax'])
            is_point_inside_bounding_box = x > bb_x_min and x < bb_x_max and y > bb_y_min and y < bb_y_max 
            
            if is_point_inside_bounding_box:
                # Draw current bounding box in image
                bb_start_point = (bb_x_min, bb_y_min)
                bb_end_point = (bb_x_max, bb_y_max)
                cv2.rectangle(image, bb_start_point, bb_end_point, box_colors_list[entry_idx], 3)
                valid_boxes_indices.append(entry_idx)

        # Regenerate the bounding box list only with valid boxes (those in which the point is contained):
        current_gt_in_list = [current_gt_in_list[valid_gt_idx] for valid_gt_idx in valid_boxes_indices]      
        
        cv2.imshow("image", image)
            
    if started_drawing_triangle and event == cv2.EVENT_MOUSEMOVE:
        new_last_point  = (x, y)
        print("Down and moving at " + "(%d, %d)" % new_last_point)

        # draw a rectangle around the region of interest
        image = clone.copy()
        cv2.rectangle(image, ref_points_list[0], new_last_point, (0, 255, 0), 2)
        cv2.imshow("image", image)

        started_drawing_triangle_and_moused_moved = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        started_drawing_triangle = False
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if started_drawing_triangle_and_moused_moved:
            ref_points_list.append((x, y))
    
            # draw a rectangle around the region of interest
            cv2.rectangle(image, ref_points_list[0], ref_points_list[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            
        started_drawing_triangle_and_moused_moved = False

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
    global image, clone, current_gt_in_list, box_colors_list

    import csv
    import sys
    from os import path
    args = parse_commandline_arguments()
    initial_frame_number = 0  # TODO: infer based on the last valid line of the output gt.csv file if any contents exist already

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
    
    csvfile_in = open(input_bounding_boxes_file, newline='')
    reader = csv.DictReader(csvfile_in)
    rows_in = list(reader)
    last_frame_number = int(rows_in[len(rows_in) - 1]['frame_number'])

    working_num_of_frames = last_frame_number + 1  # Account for zero, too
    if working_num_of_frames != len(images_list):
        print("Read number of frames does NOT match ground truth information")
        working_num_of_frames = min([working_num_of_frames, len(images_list)])
        print("Using only %d frames" % (working_num_of_frames))
        sys.exit(1)  # CHECME: exiting shouldn't be necessary
    
    # TODO: Just reading the first image
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)
    
    current_frame_number = initial_frame_number
    current_entry_index = 0
    target_frame_number = 0  # Needed for resets and rewinds
    finished_processing = False
    
    # TODO: Show image with UI instructions (lengend of keyboard commands)
    
    while not finished_processing and current_entry_index < working_num_of_frames:    
        current_gt_in_list = []
        # Extract entries for the current frame number
        current_entry_index_matches_frame_number = True
        while current_entry_index_matches_frame_number:
            current_entry_index_matches_frame_number = int(rows_in[current_entry_index]['frame_number']) == current_frame_number
            valid_entry = current_entry_index_matches_frame_number and rows_in[current_entry_index]['class_name'] == "person" 
            if valid_entry:
                current_gt_in_list.append(rows_in[current_entry_index])
            if current_entry_index_matches_frame_number:
                current_entry_index += 1
            
            # CHECKME: What if frame has no entry?
            
        if not (target_frame_number == current_frame_number):
            current_frame_number += 1  # Advance to next frame
            continue
                    
        box_colors_list = [None] * len(current_gt_in_list)
        for bb_idx, bb_color in enumerate(box_colors_list):
            color_rand = (list(np.random.choice(range(256), size=3)))  
            box_colors_list[bb_idx] = [int(color_rand[0]), int(color_rand[1]), int(color_rand[2])]
            
        print("current_entry_index", current_entry_index)    
        # keep looping until the 'q' key is pressed
        image = images_list[current_frame_number].copy()
        clone = image.copy()
        while True:
            # Draw current bounding boxes in image
            for entry_idx, gt_entry in enumerate(current_gt_in_list):
                bb_start_point = (int(gt_entry['xmin']), int(gt_entry['ymin']))
                bb_end_point = (int(gt_entry['xmax']), int(gt_entry['ymax']))
                cv2.rectangle(image, bb_start_point, bb_end_point, box_colors_list[entry_idx], 3)
            
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKeyEx(1) & 0xFF
        
            # press RIGHT arrow key or '.' to move forward to the next frame
            if key == 83 or key == ord("."):         
                break

            # press LEFT key or ',' to rewind to the previous frame
            if key == 81 or key == ord(","): 
                current_entry_index = 0  # Reset seek on input csv file
                target_frame_number = current_frame_number - 2  # Restart counting (-1 b/c it will increment next)
                current_frame_number = initial_frame_number - 1  # Restart counting  (-1 b/c it will increment next)
                break
            
            # press 'r' to reset the window
            if key == ord("r"):
                current_entry_index = 0  # Reset seek on input csv file
                target_frame_number = current_frame_number - 1  # Restart counting (-1 b/c it will increment next)
                current_frame_number = initial_frame_number - 1  # Restart counting  (-1 b/c it will increment next)
                break
            
            # press 'q' to quit
            if key == ord("q"):
                finished_processing = True
                break  # First quit the inner loop

        # Work on next frame
        current_frame_number += 1
        target_frame_number += 1  
        if target_frame_number < 0:
            target_frame_number = 0
        
    # close all open windows
    cv2.destroyAllWindows() 
    csvfile_in.close()
    
    print("Done processing input file %s" % (input_bounding_boxes_file))


if __name__ == '__main__':
    main()

