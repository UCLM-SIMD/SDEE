# SDEE

Activate the conda environment with: `conda activate sdee`.

## Slurm

- `sjstat`
- `scancel job_array`
- `squeue --job <your_job_number>`
- `squeue -u <your_user_name>`
- `sacct -u <your_user_name>`

## Aggregate features of a dataset

Given a file containing the iterations and a file containing the issues, run `python src/features_aggregator.py --iterations datasets/<filename>.csv --issues datasets/<filename>.csv --output_iterations datasets/<filename>.csv` to generate the output aggregated dataset.
