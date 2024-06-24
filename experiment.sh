#!/bin/bash

PYTHON_SCRIPT="experiment.py"

JOB_NAME="sdee"
MEM="8gb"

# Path to the file containing one numeric value per line
CONFIG_FILE="configs.txt"

# Read the number of lines (tasks) in the config file
NUM_TASKS=$(wc -l < "$CONFIG_FILE")

# Submit the job array
sbatch --job-name=$JOB_NAME --mem=$MEM --output=outputs/%A_%a.txt --error=errors/%A_%a.txt --array=1-$NUM_TASKS <<EOT
#!/bin/bash
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE)
python $PYTHON_SCRIPT \$LINE
EOT