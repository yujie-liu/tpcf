#!/bin/bash

# divide into multiple job - no dependencies
jid1=$(sbatch batch_divide.sh)

# combine into one - depend on first job to be successfully completed
jid2=$(sbatch --dependency=singleton batch_combine.sh)

