#!/bin/bash
#SBATCH --array=500-500%10
#SBATCH --exclude=worker005,worker006,worker007,worker009,worker010,worker011,worker012
#SBATCH --mem 10000                         # In MB
#SBATCH --time 2-00:00                      # Time in days-hours:min
#SBATCH --job-name=paint_baryons            # this will be displayed if you write squeue in terminal and will be in the title of all emails slurm sends
#SBATCH --requeue                           # Allow requeing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ttr@roe.ac.uk
#SBATCH -o logs/LOS_%a.out
#SBATCH -e logs/LOS_%a.err
echo node ${SLURMD_NODENAME} 1>&2

source ${HOME}/Codes/miniconda/bin/activate torch2
python -u create_lightcone.py \
--model-type=CVAE \
--CVAE-path=../trained_models/CVAE/fiducial/ \
--CGAN-module-path=../PainterGAN/painter-src/ --CGAN-parts-path=../PainterGAN/e85/ --CGAN-checkpoint=g_85_95_iter.cp \
--SLICS-base-path=/disk09/ttroester/SLICS/ --SLICS-LOS=${SLURM_ARRAY_TASK_ID} \
--n-plane=15 --tile-overlap=0.2 --output-resolution=1549 \
--drop-planes=1 \
--output-file=/disk09/ttroester/SLICS/tSZ/CVAE/y_map_${SLURM_ARRAY_TASK_ID} \
--output-file-planes=/disk09/ttroester/SLICS/tSZ/CVAE/pressure_planes_${SLURM_ARRAY_TASK_ID}.pickle
