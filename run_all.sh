#!/usr/bin/env bash
# Reproduces every number and chart (~25 min on a laptop CPU).
set -e
bash download_data.sh
python train_stage.py main 512,256 40 40 && python train_stage.py w1024 1024,512 40 40 && python train_stage.py w256 256 40 40
python teacher.py
for h in 8 16 32 64 128; do python student.py $h hard; python student.py $h distill 500; python student.py $h hard 500; done
python student.py 8 distill; python student.py 32 distill
for n in 2000 5000; do python student.py 32 hard $n; python student.py 32 distill $n; done
python student.py 32 hard 500; python student.py 32 distill 500
python make_charts.py
