#!/bin/bash

for (( counter=0; counter<5; counter++ ))
do
python air_quality_timings.py "$counter" 0 1
done
printf "\n"
