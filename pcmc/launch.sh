#!/bin/bash

#make sure pcmc-nips/lib is in the PYTHONPATH

#seed bash random number generator
RANDOM=1234

for i in {1..10}
do
	SEED=$RANDOM
	python ../pcmc/pcmc-orig-synthetic.py --trainset ../mlba/generated_N20000_seed1234.bz2 --testset ../mlba/generated_N10000_seed5678.bz2 --seed $SEED &> output$SEED.txt &
done