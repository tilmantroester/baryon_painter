#!/bin/bash

#Simulation box size
L=400.

#Omega_matter
Om_m=0.2793

#Hubble parameter
h=0.7

#Mesh size for density fields
m=512

#Number of sheets
n_sheet=8

#Models
#models+=('DMONLY_nu0_L400N1024_WMAP9')
#models+=('DMONLY_2fluid_nu0_L400N1024_WMAP9')
models+=('AGN_TUNED_nu0_v3_L400N1024_WMAP9')
#models+=('AGN_7p6_nu0_L400N1024_WMAP9')
#models+=('AGN_8p0_nu0_L400N1024_WMAP9')

#Snapshots
#snaps+=('22')
#snaps+=('26')
#snaps+=('28')
snaps+=('32')

#Input directory
indir=/data4/amead/BAHAMAS
outdir=/data/tilman/painting_baryons/BAHAMAS/M$(m)S$(n_sheet)

#Now do all the other models
for model in "${models[@]}"; do
    for snap in "${snaps[@]}"; do
	inbase=$indir/${model}_snap${snap}
	outbase=$outdir/${model}_snap${snap}
	/home/tilman/Research/painting_baryons/create_sheets/make_sheets $inbase $Om_m $h $L $m $outbase $n_sheet
    done
done
