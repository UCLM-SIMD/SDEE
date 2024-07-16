#!/bin/bash

PYTHON_SCRIPT="experiment.py"

JOB_NAME="sdee"
MEM="8gb"

# Path to the file containing one numeric value per line
python experiment_configurations.py
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

JOB_ID=$(echo $JOB_ID | awk '{print $4}')
EMAIL_SCRIPT="send_email.py"

# Submit a follow-up job to send an email after all array jobs have finished
sbatch --job-name="${JOB_NAME}_afterok" --dependency=afterok:$JOB_ID <<EOT
#!/bin/bash
python results_merger.py $JOB_ID
python $EMAIL_SCRIPT
EOT