#!/bin/bash
. load_alphafold  ## set up proper AlphaFold conda environment.

### input targets
target_lst_file=/storage/home/hcoda1/5/gmu3/data/af2ppi/git/example/test.lst
fea_dir=/storage/home/hcoda1/5/gmu3/data/af2ppi/git/example/af_fea
out_dir=/storage/home/hcoda1/5/gmu3/data/af2ppi/git/example/af_mod_test

echo "Info: input feature directory is $fea_dir"
echo "Info: result output directory is $out_dir"

af_dir=/storage/home/hcoda1/5/gmu3/data/af2ppi/git//src
cd $af_dir  || { echo 'Error: enter $af_dir failed' ; exit 1 ; }

python -u $af_dir/run_alphafold_stage2b.py \
  --target_lst_path=$target_lst_file \
  --output_dir=$out_dir \
  --feature_dir=$fea_dir \
