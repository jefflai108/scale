#!/bin/bash 

# Author: Jeff Lai 
# JHU hltcoe 2017 
# Generate wav.scp file: wav file id v.s. path to wav file 
# wav file directory: /export/b14/jlai/scale/vad/google_audioset/balanced_train/wav16khz
#
# Procedure: 
# Loop through the directory, extract the name and its pwd, write to file aduio_set_wav.scp

for filename in /export/b14/jlai/scale/vad/google_audioset/balanced_train/wav16khz/*; do 
	echo $(basename $filename .wav) $filename
done > /export/b14/jlai/scale/vad/data_prcoess/audio_set_wav.scp 

