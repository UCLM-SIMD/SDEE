#!/bin/bash

PYTHON_SCRIPT="experiment_select_percentile.py"

JOB_NAME="sdee"
MEM="8gb"

# Path to the file containing one numeric value per line
python experiment_configurations_select_percentile.py
CONFIG_FILE="configs.txt"

# Read the number of lines (tasks) in the config file
NUM_TASKS=$(wc -l < "$CONFIG_FILE")

# Submit the job array
JOB_ID=$(sbatch --job-name=$JOB_NAME --mem=$MEM --output=experiments_galgo/%A/outputs/%a.txt --error=experiments_galgo/%A/errors/%a.txt --array=1-$NUM_TASKS <<EOT
#!/bin/bash
LINE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE)
python $PYTHON_SCRIPT \$LINE
EOT
)