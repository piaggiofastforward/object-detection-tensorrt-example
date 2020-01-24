# WORK IN PROGRESS!!
# PFF Specific - Annotation : 

Clone this repository onto your local machine, and checkout VS-1361-annotation branch

## Requirements

Minimum version of Ubuntu is 18.04
Minimum version for the Nvidia Drivers on the system is 418.67 ( 430.50 also works ).
Minimum version for Docker that needs to be installed is 19.03.2

https://docs.docker.com/v17.12/install/linux/docker-ce/ubuntu/

# Add the package repositories 
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker

# next is to load the saved docker container from gita9000 into your local docker repo : 

scp <first.lastname>@10.0.0.166:/opt/annotation_template/annotation_template.tar <destination>
    
sudo docker load <destination>/annotation_template.tar

# WORK IN PROGRESS!!

# old nvisida stuff below---------------------------------------------------

# Object Detection TensorRT Example: 
This python application takes frames from a live video stream and perform object detection on GPUs. We use a pre-trained Single Shot Detection (SSD) model with Inception V2, apply TensorRT’s optimizations, generate a runtime for our GPU, and then perform inference on the video feed to get labels and bounding boxes. The application then annotates the original frames with these bounding boxes and class labels. The resulting video feed has bounding box predictions from our object detection network overlaid on it. The same approach can be extended to other tasks such as classification and segmentation.

A detailed explanation of this code sample can be found in the [How to Apply Deep Learning for Common Applications webinar](https://www.nvidia.com/en-us/about-nvidia/webinar-portal/?D2C=2003671) and also as a blog on the [NVIDIA Medium Page](https://medium.com/). 

### 1. Setup the environment

```
./setup_environment.sh
```


### 2. Run inference on the webcam

```
python SSD_Model/detect_objects_webcam.py 
```
Note: By changing the argument to the "p" flag, you can change which precision the model will run in. The options are FP32 (-p 32), FP16 (-p 16), and INT8 (-p 8)

Note: When running in INT8 precision, an extra step will be performed for calibration. But just like building an engine, it will only be performed the first time you run the model. 
