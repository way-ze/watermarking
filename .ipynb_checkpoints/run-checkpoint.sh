#!/bin/bash


# Job details
TIME=03:59:00  # HH:MM:SS (default: 04:00, max: 240:00)
NUM_GPUS=1  # GPUs per node
NUM_CPUS=1  # Number of cores (default: 1)
CPU_RAM=8192  # RAM for each core (default: 1024)
OUTFILE=test.out  # default: lsf.oJOBID
script=test.sh


# Load modules
# conda activate lm-watermarking1
module load eth_proxy


# Submit job
sbatch -–time=$TIME \
     -n $NUM_CPUS \
     --mem-per-cpu=$CPU_RAM -G $NUM_GPUS  \
     -–gpus=gtx_1080_ti:1 \  # Choices: gtx_1080_ti,rtx_3090, nvidia_a100_80gb_pcie
     # --mail-type=END,FAIL
     --output $OUTFILE \
     ./$script
