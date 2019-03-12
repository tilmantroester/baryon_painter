#!/bin/bash
#SBATCH --array=400-400%4
#SBATCH --mem 20000                         # In MB
#SBATCH --time 0-12:00                      # Time in days-hours:min
#SBATCH --job-name=paint_baryons            # this will be displayed if you write squeue in terminal and will be in the title of all emails slurm sends
#SBATCH --requeue                           # Allow requeing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ttr@roe.ac.uk
#SBATCH -o logs/LOS_%a.out
#SBATCH -e logs/LOS_%a.err
source ${HOME}/Codes/miniconda/bin/activate torch
python -u create_lightcone.py \
--model-type=CGAN \
--CVAE-path=../trained_models/CVAE/fiducial/ \
--CGAN-module-path=../PainterGAN/painter-src/ --CGAN-parts-path=../PainterGAN/e85/ --CGAN-checkpoint=g_85_95_iter.cp \
--SLICS-base-path=/disk09/ttroester/SLICS/ --SLICS-LOS=${SLURM_ARRAY_TASK_ID} \
--n-plane=2 --tile-overlap=0.2 --output-resolution=1549 \
--output-file=/disk09/ttroester/SLICS/tSZ/CGAN/y_map_${SLURM_ARRAY_TASK_ID} \
--output-file-planes=/disk09/ttroester/SLICS/tSZ/CGAN/pressure_planes_${SLURM_ARRAY_TASK_ID}.pickle
