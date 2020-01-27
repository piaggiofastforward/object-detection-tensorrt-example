# So that docker can see the webcam

programname=$0
function usage {
    echo "usage: $programname [-d {DATA}]"
    echo "  -d {DATA} runs the docker container"
    echo "            where DATA is the path to the host's data folder where the ROS bags are located."
    echo "  -h        display this help message and exit"
    exit 1
}

function run_docker {
  echo "Setting envivonment variables" 
  xhost +local:docker
  XSOCK=/tmp/.X11-unix
  XAUTH=/tmp/.docker.xauth
  xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

  # Populate the rest of the default arguments
  common_args="--gpus all -it -v `pwd`:/mnt --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH "

  # Start the docker container
  echo "Starting docker container" 

  # Run container on a x86_64 device
  docker run \
    $common_args \
    $extra_args \
    annotation_template

  exit 0
}

# Configure using arguements given to this script
extra_args=""
while getopts "hd:" OPTION; do
  case $OPTION in
  d)
    echo "Adding extra options provided"
  
    extra_args="$extra_args -v $OPTARG:/media "
    run_docker
    ;;
  h)
    usage
    ;;
  *)
    echo "Incorrect options provided"
    usage
    exit 1
    ;;
  esac
  
done

# Just run docker function as the last resort
run_docker 

