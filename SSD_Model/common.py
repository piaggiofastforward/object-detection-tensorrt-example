# /// Copyright (c) 2020 Piaggio Fast Forward (PFF), Inc.
# /// All Rights Reserved. Reproduction, publication,
# /// or re-transmittal not allowed without written permission of PFF, Inc.

from __future__ import division
from __future__ import print_function

# import the necessary packages
import cv2
import numpy as np


def rgb2bgr_color(rgb_color):
    return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))


def draw_points(img_input, points_uv_coords, num_valid_points=None, color_rgb=None, thickness=1):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param points_uv_coords: 
    @note: the uv coordinates list or ndarray must be of shape (n, 2) for n points.
    @note: the coordinates will be expressed as integers while visualizing
    
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
            try:
                pt_as_tuple = (int(pt[0]), int(pt[1]))  # Recall: (pt[0],pt[1]) # (x, u or col and y, v or row)
                cv2.circle(img_input, pt_as_tuple, 2, color_rgb, thickness, 8, 0)
            except:
                print("Point", pt_as_tuple, "cannot be drawn!")


def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    '''
    Custom drawing of lines with support for dotted style
    '''
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in  np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def draw_poly(img, pts, color, thickness=1, style='dotted',):
    '''
    Custom drawing of polygons with support for dotted-edge style
    '''
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_line(img, s, e, color, thickness, style)


def draw_rectangle(img, pt1, pt2, color, thickness=1, style='dotted'):
    '''
    Custom drawing of rectangles with support for dotted-edge style
    '''
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])] 
    draw_poly(img, pts, color, thickness, style)

    
def draw_instructions_window(win_name, instruction_text_lines):
    '''
    Show image with UI instructions (lengend of mouse and keyboard commands)
    
    @param instruction_text_lines: A list of strings 
    '''
    label_height = 50
    
    label_width = 0
    for instruction_text in instruction_text_lines:
        if len(instruction_text) > label_width:
            label_width = len(instruction_text)
    
    label_width = int(label_width * label_height / 2.5)  # Scale width

    num_lines_text = len(instruction_text_lines)
    image = np.ones([label_height * num_lines_text, label_width, 3], np.uint8) * 128  # The multiplier value is for painting a grayish color
    
    for idx, instruction_text in enumerate(instruction_text_lines):
        cv2.rectangle(image, (0, idx * label_height), (label_width, (idx + 1) * label_height), (255, 255, 255))
        cv2.putText(image, instruction_text, (0, int((idx + 0.5) * label_height)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
    
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, image)
    cv2.waitKey(1)


def str2bool(v):
    # Used for the argsparser to simulate a boolean type
    return v.lower() in ("yes", "true", "t", "on", "1")


def resolve_path(input_path):
    from os import path, getcwd
    if input_path is None or len(input_path) == 0:
        input_path = getcwd()
    if path.isfile(input_path):  # Do this in case a filename was passed instead
        path_of_interest = path.dirname(input_path)  
    else:
        path_of_interest = input_path
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
    img_names.sort()  # Names will be sorted ascendengly

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
                    root_path, name, ext = splitfn(fn)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, img)
            else:
                print("failed!")
        except:
            warnings.warn("Warning...image index %d not found at %s" % (i, __name__))

    if show_images:
        cv2.waitKey(1)

    return images  # all even if None

def get_bgr_colors_list(num_colors = 10, do_random = False):
    max_num_predifined_colors = 10
    colors_list_bgr = []
    if do_random == False:
        # Predifined set of RGB colors
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        predifined_rgb_colors = [None] * max_num_predifined_colors
        predifined_rgb_colors[0] = (255, 0, 0)  # Red     #FF0000     (255,0,0)
        predifined_rgb_colors[1] = (0, 0, 255)  # Blue     #0000FF     (0,0,255)
        predifined_rgb_colors[2] =  (245,222,179)  # wheat     #F5DEB3     (245,222,179)
        predifined_rgb_colors[3] = (0, 255, 255)  # Cyan / Aqua     #00FFFF     (0,255,255)
        predifined_rgb_colors[4] = (255, 0, 255)  # Magenta / Fuchsia     #FF00FF     (255,0,255)
        predifined_rgb_colors[5] = (255, 165, 0)  # orange     #FFA500     (255,165,0)
        predifined_rgb_colors[6] = (128, 0, 0)  # Maroon     #800000     (128,0,0)
        predifined_rgb_colors[7] = (128, 128, 0)  # Olive     #808000     (128,128,0)
        predifined_rgb_colors[8] = (128, 0, 128)  # Purple     #800080     (128,0,128)
        predifined_rgb_colors[9] = (0, 128, 128)  # Teal     #008080     (0,128,128)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        for idx_rgb in range(min(max_num_predifined_colors, num_colors)):
            # NOTE: we need to convert RGB to BGR
            colors_list_bgr.append(rgb2bgr_color(predifined_rgb_colors[idx_rgb]))
            
        if num_colors > max_num_predifined_colors:
            # Append remaining colors randomly
            do_random = True
        
    if do_random:    
        for idx in range(num_colors - len(colors_list_bgr)):
            color_rand = (list(np.random.choice(range(256), size=3)))  
            colors_list_bgr.append((int(color_rand[0]), int(color_rand[1]), int(color_rand[2])))
    
    return colors_list_bgr  


def save_image_sequence_from_video(video_input_fn, images_path):
    from os import path
    video_file_name = path.realpath(path.expanduser(video_input_fn))
    images_path = make_sure_path_exists(images_path)
    prefix_name_templated = "{seq:06d}.png"
    output_img_name_templated = path.join(images_path, prefix_name_templated)
    
    # Path to video file 
    video_obj = cv2.VideoCapture(video_file_name) 

    win_name = "Frame Saved"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
      
    # Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1
    while success: 
        # extract frames 
        success, image = video_obj.read() 
        if success:
            img_name = output_img_name_templated.format(seq=count)
            # Saves the frames with frame-count 
            cv2.imwrite(img_name, image) 
            cv2.imshow(win_name, image)
            cv2.waitKey(1)
            
            count += 1
        
    print("Saved {count:d} images to {output_path:s}".format(count=count, output_path=images_path))    
