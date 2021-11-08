#!/bin/bash

# You need to provide DATA_DIR and load_alphafold
DATA_DIR=$HOME/scratch/afold/data ## change this to point to alphafold's DL parameter directory
if [[ -r load_alphafold ]]; then
  . load_alphafold  ## set up proper AlphaFold conda environment.
fi

### input targets
target_lst_file=test.lst
fea_dir=af_fea
out_dir=af_mod

model=model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm
preset=super

recycling_setting=1

echo "Info: input feature directory is $fea_dir"
echo "Info: result output directory is $out_dir"

af_dir=../src

python -u $af_dir/run_alphafold_stage2a_comp.py --target_lst_path=$target_lst_file \
  --data_dir=$DATA_DIR --output_dir=$out_dir --feature_dir=$fea_dir \
  --model_names=$model \
  --preset=$preset \
  --save_recycled=$recycling_setting
