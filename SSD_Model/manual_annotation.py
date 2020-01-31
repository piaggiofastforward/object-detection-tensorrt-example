# /// Copyright (c) 2020 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

from __future__ import division
from __future__ import print_function

# import the necessary packages
import cv2
import numpy as np
import argparse

from common import draw_instructions_window, draw_points, draw_rectangle
from common import get_images, save_image_sequence_from_video, get_bgr_colors_list, verify_path, make_sure_path_exists

# Initialize the shared lists
ref_points_list = []  # reference points
current_gt_in_list = []  # current ground truth entries
box_colors_list = []
current_validated_bounding_box = []  # It will contain [x_min, y_min, width, height] of the bounding box 

started_drawing_triangle = False
started_drawing_triangle_and_moused_moved = False
image = None
clone = None

instruction_text_lines = ["MOUSE EVENTS:",
                          "  Right Click inside box to validate bounded target",
                          "  Right Click outside of box if target does not exist",
                          "  Draw bounding box around target to correct",
                          "     NOTE: yellow dotted box refers to previous frame.",
                          " ",
                          "KEYBOARD CONTROL:",
                          "  Next frame:     Rigth arrow (>) or Space bar (\___/)",
                          "  Previous frame: Left arrow (<)",
                          "  Reset frame:    'r'",
                          "  Quit sequence:  'q'"
                          ]

            
def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(
        description='Run object detection inference on input image.')
    
    parser.add_argument('-i', '--images_input_path', help='Path to folder containing images (or where images from video will be put)', type=str, required=True)
    parser.add_argument('-v', '--video_input', help='Path to a video file instead of images sequence', type=str)
    parser.add_argument('-a', '--auto_annotation', help='The automatic annnotation filename containing the bounding box information as well as frame number', type=str, required=True)
    parser.add_argument('-o', '--output', help='Complete name for the resulting annotation file', type=str)

    # Parse arguments passed
    args = parser.parse_args()
    return args


def reorder_bounding_box_corners(points_pairs):
    '''
    Resolve points for Top-Left corner and then Bottom-Right
    
    @param points_pairs: A list of 2 points with (u,v) coordinates each
    '''
    from sys import maxsize
    u_min = maxsize
    v_min = maxsize
    u_max = 0
    v_max = 0            
    for pt_coord in points_pairs:
        u, v = pt_coord
        if u <= u_min:
            u_min = u
        if v <= v_min:
            v_min = v
        if u >= u_max:
            u_max = u
        if v >= v_max:
            v_max = v

    # Form resulting list of point coordinates
    bb_corners = [(u_min, v_min), (u_max, v_max)]
    return bb_corners


def fill_in_bounding_box_selected(x_min, y_min, x_max, y_max):
    '''
    @return a list of the bounding box parameters: [x_min, y_min, width, height]
    '''
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    
    bounding_box_entry = [x_min, y_min, width, height]
    
    return bounding_box_entry


def get_centroid(x, y, width, height):
    centroid_x_coord = x + width / 2
    centroid_y_coord = y + height / 2
    return (centroid_x_coord, centroid_y_coord)


def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_points_list, current_gt_in_list, started_drawing_triangle, started_drawing_triangle_and_moused_moved, current_validated_bounding_box, image, clone

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    
    if event == cv2.EVENT_RBUTTONDOWN:
        started_drawing_triangle = False
        started_drawing_triangle_and_moused_moved = False
        current_validated_bounding_box = []
        ref_points_list = [(x, y)]
        
        # draw this point
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
        # CHECKME: Assuming only one box exist in the current_gt_in_list
        for valid_entry in current_gt_in_list:
            current_validated_bounding_box = fill_in_bounding_box_selected(x_min=int(valid_entry['xmin']), y_min=int(valid_entry['ymin']), x_max=int(valid_entry['xmax']), y_max=int(valid_entry['ymax']))

        cv2.imshow("image", image)

    if event == cv2.EVENT_LBUTTONDOWN:
        started_drawing_triangle = True
        started_drawing_triangle_and_moused_moved = False
        current_validated_bounding_box = []
        ref_points_list = [(x, y)]
        cv2.imshow("image", image)
            
    if started_drawing_triangle and event == cv2.EVENT_MOUSEMOVE:
        current_validated_bounding_box = []
        current_gt_in_list = []  # Clear up old boxes 
        new_last_point  = (x, y)
        # print("Down and moving at " + "(%d, %d)" % new_last_point)

        # draw a rectangle around the region of interest
        image = clone.copy()
        cv2.rectangle(image, ref_points_list[0], new_last_point, (0, 255, 0), 2)
        cv2.imshow("image", image)

        started_drawing_triangle_and_moused_moved = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP and started_drawing_triangle_and_moused_moved:
        started_drawing_triangle = False
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        if started_drawing_triangle_and_moused_moved:
            ref_points_list.append((x, y))
            # Resolve points for Top-Left corner and then Bottom-Right
            ref_points_list = reorder_bounding_box_corners(ref_points_list)
            current_validated_bounding_box = fill_in_bounding_box_selected(x_min=ref_points_list[0][0], y_min=ref_points_list[0][1], x_max=ref_points_list[1][0], y_max=ref_points_list[1][1])
            # draw a rectangle around the region of interest
            cv2.rectangle(image, ref_points_list[0], ref_points_list[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            
        started_drawing_triangle_and_moused_moved = False

    # Paths correctness verifier


def annotate(images_path, input_bounding_boxes_filename, output_bounding_boxes_filename):
    '''
    The manual annotation main function
    
    @param images_path: The complete path to the folder with the sequence of images
    @param input_bounding_boxes_filename: The complete name of the file containing the automatic bounding boxes information
    @param output_bounding_boxes_filename: The complete name of the file that will store the validated bounding boxes
    '''
    global image, clone, current_gt_in_list, box_colors_list, current_validated_bounding_box

    import csv
    import sys
    from os import path, stat
    
    # Get file names of image sequence
    input_img_folder = verify_path(images_path)  # This must now exist at this point!
    input_imgs_template = path.join(input_img_folder, "*")
    image_names_list = get_images(filename_template=input_imgs_template, indices_list=[], show_images=False, return_names_only=True)
    
    csvfile_in = open(input_bounding_boxes_filename, newline='')
    reader = csv.DictReader(csvfile_in)
    rows_auto_gt_input = list(reader)
    last_frame_number = int(rows_auto_gt_input[len(rows_auto_gt_input) - 1]['frame_number'])

    working_num_of_frames = last_frame_number + 1  # Account for zero, too
    if working_num_of_frames != len(image_names_list):
        print("Read number of frames does NOT match ground truth information")
        working_num_of_frames = len(image_names_list)
        print("Using all %d frames" % (working_num_of_frames))

    # Start with an array full of NANs
    output_array_of_bounding_boxes = np.ndarray((0, 4), dtype='int')
    if path.isfile(output_bounding_boxes_filename):
        # File already exist, so read contents and close 
        # Check if file is empty
        if not stat(output_bounding_boxes_filename).st_size == 0:
            output_array_of_bounding_boxes_temp = np.genfromtxt(output_bounding_boxes_filename, delimiter=',', dtype='float').reshape(-1, 4)
            # Reduce the array till the last non-NANs in the sequence file
            if len(output_array_of_bounding_boxes_temp):
                non_nan_indices = np.argwhere(np.logical_not(np.any(np.isnan(output_array_of_bounding_boxes_temp), axis=1)))
                last_non_nan_index = non_nan_indices[-1, 0]
                output_array_of_bounding_boxes = output_array_of_bounding_boxes_temp[:last_non_nan_index + 1]
        
    # Infer based on the last valid line of the output gt.csv file if any contents exist already
    initial_frame_number = len(output_array_of_bounding_boxes)  

    # Append the remaining of the file with NANs
    output_array_of_bounding_boxes_remaining = np.ndarray((working_num_of_frames - initial_frame_number, 4), dtype='int') * np.nan
    output_array_of_bounding_boxes = np.vstack((output_array_of_bounding_boxes, output_array_of_bounding_boxes_remaining))

    # WISH: Use the confident level to tune down the color of the box                    
    box_colors_list = get_bgr_colors_list(num_colors=5, do_random=False)

    draw_instructions_window(win_name="INSTRUCTIONS", instruction_text_lines=instruction_text_lines)
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", shape_selection)
    
    # When we are starting on the last frame, we may not want to be done immediately
    if initial_frame_number >= working_num_of_frames - 1: 
        initial_frame_number = initial_frame_number - 1
        
    current_frame_number = initial_frame_number
    target_frame_number = initial_frame_number  # Needed for resets and rewinds
    previous_base_entry_indices = [0]  
    finished_processing = False
    highest_visited_frame_number = target_frame_number - 1
    
    while not finished_processing and current_frame_number < working_num_of_frames:    
        current_gt_in_list = []
        current_validated_bounding_box = []
        current_entry_index_matches_frame_number = False
        current_base_entry_index = previous_base_entry_indices[-1]
        current_entry_index = current_base_entry_index  # Always restart the search onto the target frame
        print("TARGET frame number:", target_frame_number)
        
        # Find the first entry index matching the current frame number
        while not current_entry_index_matches_frame_number and current_entry_index < len(rows_auto_gt_input):
            current_entry_index_matches_frame_number = int(rows_auto_gt_input[current_entry_index]['frame_number']) == current_frame_number
            if current_entry_index_matches_frame_number:
                current_base_entry_index = current_entry_index
                previous_base_entry_indices.append(current_base_entry_index)
                # print("Added index %d to stack" % (current_base_entry_index))
                # print("Stack currently has ", previous_base_entry_indices)
            else:
                if current_entry_index < len(rows_auto_gt_input):
                    current_entry_index += 1  # Keep searching for the index until a first match is found
                else:
                    break
                
        # Extract entries for the current frame number
        while current_entry_index_matches_frame_number and current_entry_index < len(rows_auto_gt_input):
            current_entry_index_matches_frame_number = int(rows_auto_gt_input[current_entry_index]['frame_number']) == current_frame_number
            valid_entry = current_entry_index_matches_frame_number and rows_auto_gt_input[current_entry_index]['class_name'] == "person" 
            if valid_entry:
                current_gt_in_list.append(rows_auto_gt_input[current_entry_index])
            if current_entry_index_matches_frame_number:
                if current_entry_index < len(rows_auto_gt_input):
                    current_entry_index += 1  # Keep comparing the next entry
                else:
                    break
            
        if not np.any(np.isnan(output_array_of_bounding_boxes[current_frame_number]), axis=0):
            current_gt_in_list = []
            current_validated_bounding_box = output_array_of_bounding_boxes[current_frame_number]
            
        if not (target_frame_number == current_frame_number):
            current_frame_number += 1  # Advance to next frame
            continue

        image = cv2.imread(image_names_list[target_frame_number], cv2.IMREAD_COLOR)
        clone = image.copy()
        while True:
            # Draw last validated bounding box as a yellow dotted-edge rectangle    
            if target_frame_number > 0 and not np.any(np.isnan(output_array_of_bounding_boxes[target_frame_number - 1]), axis=0):
                prev_bb_start_point = (int(output_array_of_bounding_boxes[target_frame_number - 1][0]), int(output_array_of_bounding_boxes[target_frame_number - 1][1]))
                prev_bb_end_point = (int(prev_bb_start_point[0] + output_array_of_bounding_boxes[target_frame_number - 1][2]), int(prev_bb_start_point[1] + output_array_of_bounding_boxes[target_frame_number - 1][3]))
                draw_rectangle(image, pt1=prev_bb_start_point, pt2=prev_bb_end_point, color=(0, 255, 255), thickness=2, style='dotted')

            # Draw current bounding boxes in image
            if len(current_validated_bounding_box) > 0:
                bb_start_point = (int(current_validated_bounding_box[0]), int(current_validated_bounding_box[1]))
                bb_end_point = (int(current_validated_bounding_box[0] + current_validated_bounding_box[2]), int(current_validated_bounding_box[1] + current_validated_bounding_box[3]))
                cv2.rectangle(image, bb_start_point, bb_end_point, (0, 255, 0), 3)  # Always draw this a "green"
            else:
                # Pick the closest and similar in size bounding box stablished in the previous frame
                if current_frame_number > 0 and not np.any(np.isnan(output_array_of_bounding_boxes[current_frame_number - 1]), axis=0):
                    previous_gt_centroid = get_centroid(*output_array_of_bounding_boxes[current_frame_number - 1])
                    smallest_centroids_distance = np.inf
                    for valid_entry in current_gt_in_list:
                        candidate_bounding_box = fill_in_bounding_box_selected(x_min=int(valid_entry['xmin']), y_min=int(valid_entry['ymin']), x_max=int(valid_entry['xmax']), y_max=int(valid_entry['ymax']))
                        candidate_gt_centroid = get_centroid(*candidate_bounding_box)
                        # measure distances (L2 norm) between centroids
                        current_centroids_distance = ((previous_gt_centroid[0] - candidate_gt_centroid[0]) ** 2 + (previous_gt_centroid[1] - candidate_gt_centroid[1]) ** 2) ** 0.5
                        if current_centroids_distance < smallest_centroids_distance:
                            smallest_centroids_distance = current_centroids_distance
                            current_validated_bounding_box = candidate_bounding_box
                            # TODO: it would be helpful to validate also by the size similarity of the bounding boxes
                            # However, its becoming too heuristic!
                else:
                    # Pick box with highest confidence
                    highest_confidence = 0.
                    for valid_entry in current_gt_in_list:
                        current_confidence = float(valid_entry['confidence'][:-1])  # We need to remove the percent sign
                        if current_confidence >= highest_confidence:
                            highest_confidence = current_confidence
                            current_validated_bounding_box = fill_in_bounding_box_selected(x_min=int(valid_entry['xmin']), y_min=int(valid_entry['ymin']), x_max=int(valid_entry['xmax']), y_max=int(valid_entry['ymax']))
                
                for entry_idx, gt_entry in enumerate(current_gt_in_list):
                    bb_start_point = (int(gt_entry['xmin']), int(gt_entry['ymin']))
                    bb_end_point = (int(gt_entry['xmax']), int(gt_entry['ymax']))
                    cv2.rectangle(image, bb_start_point, bb_end_point, box_colors_list[entry_idx], 3)

            
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKeyEx(1) & 0xFF

            # press SPACE bar or RIGHT arrow key or '.' to move forward to the next frame (also saves to file)
            # NOTE: we also handle the LEFT arrow or ',' case here
            if key == ord(" ") or key == 83 or key == ord(".") or key == 81 or key == ord(","):         
                if highest_visited_frame_number <= target_frame_number:
                    highest_visited_frame_number = target_frame_number
                
                # Save current validated bounding box to file
                if len(current_validated_bounding_box) == 0:
                    output_array_of_bounding_boxes[target_frame_number] = [np.nan, np.nan, np.nan, np.nan]
                else:
                    output_array_of_bounding_boxes[target_frame_number] = current_validated_bounding_box
                print("Frame %06d saving BB: %s" % (target_frame_number, output_array_of_bounding_boxes[target_frame_number]))  
                
                # press LEFT key or ',' to rewind to the previous frame
                if key == 81 or key == ord(","): 
                    # print("REWIND:")
                    if current_frame_number > 0:
                        if len(previous_base_entry_indices) > 2:
                            removed_index = previous_base_entry_indices.pop()  # Removed this recently appended base index
                            # print("Removed index %d from stack" % (removed_index))
                        if len(previous_base_entry_indices) > 1:
                            removed_index = previous_base_entry_indices.pop()  # Removed the previously appended index
                            # print("as well as index %d from stack" % (removed_index))
                        # print("Stack currently has ", previous_base_entry_indices)
                        target_frame_number = target_frame_number - 2  # Restart counting (-1 b/c it will increment next)
                        current_frame_number = target_frame_number 
                    else:
                        target_frame_number = target_frame_number - 1  # Restart counting (-1 b/c it will increment next)
                        current_frame_number = target_frame_number
                break
            # press 'r' to reset the window
            elif key == ord("r"):
                # print("RESET:")
                if len(previous_base_entry_indices) > 1:
                    removed_index = previous_base_entry_indices.pop()  # Removed this recently appended base index
                    # print("Removed index %d from stack" % (removed_index))
                # print("Stack currently has ", previous_base_entry_indices)
                output_array_of_bounding_boxes[target_frame_number] = [np.nan, np.nan, np.nan, np.nan]
                target_frame_number = target_frame_number - 1  # Restart counting (-1 b/c it will increment next)
                current_frame_number = target_frame_number
                break
            # press 'q' to quit
            elif key == ord("q"):
                finished_processing = True
                break  # First quit the inner loop
            
        # Work on next frame
        current_frame_number += 1
        target_frame_number += 1  
        if target_frame_number < 0:
            target_frame_number = 0
        
    # Write results of output_array_of_bounding_boxes to file
    # NOTE: only saving until the highest visited frame
    np.savetxt(output_bounding_boxes_filename, output_array_of_bounding_boxes[:highest_visited_frame_number + 1], delimiter=',', fmt='%.0f')
        
    # close all open windows
    cv2.destroyAllWindows() 
    csvfile_in.close()
    
    print("Done processing input file %s [highest validated frame = %06d]" % (input_bounding_boxes_filename, highest_visited_frame_number))


def main():
    import sys
    from os import path
    args = parse_commandline_arguments()

    if not path.isfile(args.auto_annotation):
        print("Input bounding boxes (labels) file is missing!")
        sys.exit(1)
    input_bounding_boxes_filename = args.auto_annotation
        
    output_folder, output_fn_only = path.split(args.output)
    output_folder = make_sure_path_exists(output_folder)
    output_bounding_boxes_filename = path.join(output_folder, output_fn_only)

    images_path = args.images_input_path

    # Giving precedence to any provided input video file, which will be split into images in the path provided.
    if not args.video_input is None:  
        save_image_sequence_from_video(video_input_fn=args.video_input, images_path=images_path)

    annotate(images_path, input_bounding_boxes_filename, output_bounding_boxes_filename)    
    

if __name__ == '__main__':
    main()

