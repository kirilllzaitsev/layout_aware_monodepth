#!/bin/bash

cd ../layout_aware_monodepth || exit 1

python train_prototyping.py "$@"