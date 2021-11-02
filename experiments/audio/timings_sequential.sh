#!/bin/bash

for (( counter=0; counter<4; counter++ ))
do
python3 audio_timings.py 2 0 7 "$counter"
done
printf "\n"
