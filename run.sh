#!/bin/bash

python3 demo.py \
    --input "toddler-2675884_640.jpg" \
    --output "output" \
    --detector-weights "models/yolov8x_person_face.pt" \
    --checkpoint "models/mivolo_imbd.pth.tar" \
    --device "cpu"
    
    
    
#    --with-persons
#    --draw
