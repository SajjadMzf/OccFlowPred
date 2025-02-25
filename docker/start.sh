#!/bin/bash

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=ofmpnet)" ]; then
    docker rm ofmpnet;
fi

docker run -it -d --rm \
    --gpus all \
    --net host \
    --device=/dev/sdb \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="40g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name ofmpnet \
    -v $workspace_dir/:/home/user/ofmpnet/:rw \
    -v /media/wmg-5gcat/ssd-roger/Waymo_Dataset:/home/user/ofmpnet/Waymo_Dataset/:rw \
    x64/ofmpnet:latest
    

docker exec -it --user "root" ofmpnet \
    /bin/bash -c "
    export PYTHONPATH=\"${PYTHONPATH}:/home/user/ofmpnet\";
    cd /home/user/ofmpnet
    "