#!/bin/bash

PYTHON_SCRIPT="experiment.py"

JOB_NAME="sdee"
OUTPUT_FILE="outputs/%j/%a.txt"
ERROR_FILE="errors/%j/%a.txt"
MEM="8gb"

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_FILE
#SBATCH --error=$ERROR_FILE
#SBATCH --mem=$MEM

python $PYTHON_SCRIPT
EOT