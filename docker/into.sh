#!/bin/bash
docker exec -u 0 -it ofmpnet \
    /bin/bash -c "
    export PYTHONPATH=\"${PYTHONPATH}:/home/user/ofmpnet\";
    cd /home/user/ofmpnet;
    /bin/bash"

