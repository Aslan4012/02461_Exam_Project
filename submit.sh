#!/bin/bash
#BSUB -J Train_CNN          # Job name
#BSUB -q gpua100            # Queue
#BSUB -n 8                  # Request 8 cores
#BSUB -R "span[hosts=1]"    # Ensure all cores are on the same host
#BSUB -R "rusage[mem=16GB]" # Request 16 GB of memory per core (8 × 16 = 128 GB total)
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 A100 GPU
#BSUB -M 20GB               # Memory limit
#BSUB -W 24:00              # Wall clock time (24 hours)
#BSUB -u aslan11@icloud.com # Email
#BSUB -B                    # Send email when the job starts
#BSUB -N                    # Send email when the job finishes
#BSUB -o Output_%J.out      # Output file
#BSUB -e Error_%J.err       # Error file

# Load necessary modules (if any)

# Activate the environment
source ~/work/s224819/anaconda3/bin/activate
conda activate 02461_Exam_Env

# Navigate to project directory
cd ~/work3/s224819/02461_Exam_Project

# Run Python script
python3 -u CNN.py