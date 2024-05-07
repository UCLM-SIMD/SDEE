#!/bin/bash

source activate sdee

NAME="vperez-sdee"

qsub -N "$NAME" \
        -e "$PWD"/errors/ \
        -o "$PWD"/outputs/ \
        -v CWD="$PWD" \
        "$PWD"/experiment_job.sh
done