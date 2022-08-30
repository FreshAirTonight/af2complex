#!/bin/bash

### input targets
target_lst_file=targets/test_iscore.lst
fea_dir=af2c_fea
out_dir=af2c_mod

echo "Info: input feature directory is $fea_dir"
echo "Info: result output directory is $out_dir"

af_dir=../src

cluster_edge_thres=10

python -u ../tools/run_interface_score.py \
  --target_lst_path=$target_lst_file \
  --output_dir=$out_dir \
  --feature_dir=$fea_dir \
  --do_cluster_analysis \
  --cluster_edge_thres=$cluster_edge_thres \
