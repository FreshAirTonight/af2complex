#!/bin/bash

# You need to provide DATA_DIR and load_alphafold
DATA_DIR=$HOME/data/af2ppi/git
if [[ -r load_alphafold ]]; then
  . load_alphafold  ## set up proper AlphaFold conda environment.
fi

### input targets
target_lst_file="$DATA_DIR/example/test.lst"
fea_dir="$DATA_DIR/example/af_fea"
out_dir="$DATA_DIR/example/af_mod_test"

echo "Info: input feature directory is $fea_dir"
echo "Info: result output directory is $out_dir"

af_dir="$DATA_DIR/src"
cd "$af_dir" || { echo 'Error: enter $af_dir failed' ; exit 1 ; }
mkdir -p "$out_dir" || { echo 'Error: mkdir $out_dir failed' ; exit 1 ; }

python -u "$af_dir/run_alphafold_stage2b.py" \
  --target_lst_path="$target_lst_file" \
  --output_dir="$out_dir" \
  --feature_dir="$fea_dir"
