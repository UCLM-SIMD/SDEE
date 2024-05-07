#!/bin/sh

#PBS -l mem=8gb

# Make sure that the script is run
# in the current working directory
cd $CWD

source activate sdee

python "$FILE"
