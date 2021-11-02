#!/bin/bash

for (( counter=0; counter<4; counter++ ))
do
python3 audio_timings.py 2 1 7 "$counter"
done
printf "\n"
