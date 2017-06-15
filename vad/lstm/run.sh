#!/bin/bash
. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc_1

#steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" `pwd`/train log/mfcc_log $mfccdir 

steps/mfcc_cmv_ark2h5.sh train 10 ./mfcc_cmvn.h5 exp/ark2h5
