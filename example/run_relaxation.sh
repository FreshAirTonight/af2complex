#!/bin/bash

# You need to provide load_alphafold to set up correct enviroment
if [[ -r $HOME/bin/load_alphafold  ]]; then
  . load_alphafold  ## set up proper AlphaFold conda environment.
fi

af_dir="../src"
cd "$af_dir" || { echo 'Error: enter $af_dir failed' ; exit 1 ; }


### input targets
target_lst_file="../example/test.lst"
fea_dir="../example/af_fea"
out_dir="../example/af_mod"

echo "Info: searching for model_*.pdb files under $out_dir for minimization"

python -u "./run_alphafold_stage2b.py" \
  --target_lst_path="$target_lst_file" \
  --output_dir="$out_dir" \
  --feature_dir="$fea_dir"
