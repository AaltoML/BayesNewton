#!/bin/bash

for (( counter=0; counter<6; counter++ ))
do
python3 rainforest_timings.py 0 "$counter" 0 0
done
printf "\n"
