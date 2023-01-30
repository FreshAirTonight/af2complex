#!/bin/bash
#
# Run examples 1 to 3 by specifying DL model, model preset, and MSA mode in a
# target list file <examples.lst>

# You need to take care of these two items, which are dependent on your installation.
# 1) activate your conda environment for AlphaFold if you use conda
# . load_alphafold
# 2) change this to point to alphafold's deep learning model parameter directory
DATA_DIR=$HOME/scratch/afold/data

### input targets
target_lst_file=targets/examples.lst  # a list of target with stoichiometry
fea_dir=af2c_fea   # directory to input feature pickle files
out_dir=af2c_mod # model output directory, s.t. output files will be on $out_dir/$target

### run preset, note this is different from model_preset defined below
### This preset defined the number of recycles, ensembles, MSA cluster sizes (for monomer_ptm models)
preset=economy # up to 3 recycles, 1 ensemble.

# these two options can be overwritten in examples.lst
model=model_1_multimer_v3
model_preset=multimer_np


recycling_setting=1   # output metrics but not saving pdb files during intermediate recycles

echo "Info: input feature directory is $fea_dir"
echo "Info: result output directory is $out_dir"
echo "Info: model preset is $model_preset"

# AF2Complex source code directory
af_dir=../src

# use cuda memory unification for a large target
#export TF_FORCE_UNIFIED_MEMORY=1
#export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0


python -u $af_dir/run_af2c_mod.py --target_lst_path=$target_lst_file \
  --data_dir=$DATA_DIR --output_dir=$out_dir --feature_dir=$fea_dir \
  --model_names=$model \
  --preset=$preset \
  --model_preset=$model_preset \
  --save_recycled=$recycling_setting \
