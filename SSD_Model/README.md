# Annotation Tools 

These tools facilitate automatic annotation (via on the Inception DNN) and manual annotation tools for validation of
the PFF dataset.

## Bag Exploder Tool

### Requirements

- Path to the PFF-vision `decoder` binary

### Usage

## Automatic Annotation Tool

### Requirements

This tool should be run within the [dataset_tools_docker Docker container](TODO: link is missing)

### Usage

Run the Docker via the helper script

    $ sudo ./run_environment.sh -d PATH_TO_MY_LOCAL_DATA_FOLDER

where the optional argument `-d PATH_TO_MY_LOCAL_DATA_FOLDER` allows mounting the volume of this data path to the container's `/media` folder

Run the annotation tool TODO

## Manual Annotation tool

This tool helps validate existing automatically-annotated (labelled) sequences.

### Requirements

- Python 3+

- OpenCV 3+ with Python3 bindings

### Usage

The single `manual_annotation.py` script can be run as follows:

```
required arguments:
  -i IMAGES_INPUT_PATH, --images_input_path IMAGES_INPUT_PATH
                        Path to folder containing images (or where images from
                        video will be put)
  -a AUTO_ANNOTATION, --auto_annotation AUTO_ANNOTATION
                        The automatic annnotation filename containing the
                        bounding box information as well as frame number
  -o OUTPUT, --output OUTPUT
                        Complete name for the resulting annotation file

optional arguments:
  -v VIDEO_INPUT, --video_input VIDEO_INPUT
```

For example, 

    $ python3 manual_annotation.py -i /media/dataset/sequences/log-2020-01-30-21-33-26/rs430/gray -a /media/dataset/groundtruth/log-2020-01-30-21-33-26/rs430/gray/automatic_boundingbox_log-2020-01-30-21-33-26_gray.csv -o /media/dataset/groundtruth/log-2020-01-30-21-33-26/rs430/gray/boundingbox_log-2020-01-30-21-33-26_gray.csv

