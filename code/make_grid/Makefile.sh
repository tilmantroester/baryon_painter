#!/bin/sh

export LD_LIBRARY_PATH=/home/tilman/lib/gcc/lib64/:$LD_LIBRARY_PATH
export PATH=/home/tilman/lib/gcc/bin:$PATH

gfortran -ffree-line-length-none -fimplicit-none -std=gnu -c -g constants.f90 BAHAMAS_sheets.f90 
gfortran *.o -o make_sheets