#!/bin/bash
#BSUB -J Train_CNN          # Job name
#BSUB -q hpc                # Queue name
#BSUB -n 8                  # Request 8 cores
#BSUB -R "span[hosts=1]"    # Ensure all cores are on the same host
#BSUB -R "rusage[mem=16GB]" # Request 16 GB of memory per core (8 × 16 = 128 GB total)
#BSUB -M 16GB               # Memory limit
#BSUB -W 24:00              # Wall clock time (24 hours)
#BSUB -u aslan11@icloud.com # Email
#BSUB -B                    # Send email when the job starts
#BSUB -N                    # Send email when the job finishes
#BSUB -Ne                  # Send email when the job is aborted
#BSUB -o Output_%J.out      # Output file
#BSUB -e Error_%J.err       # Error file

# Load necessary modules
module load python/3.11.7

# Activate virtual environment (if applicable)
source /Users/aslandalhoffbehbahani/anaconda3/envs/ReinforcementLearning

# Navigate to project directory
cd /Users/aslandalhoffbehbahani/Documents/02461_Exam_Project

# Run Python script
python3 -u CNN.py
