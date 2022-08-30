"""Adds piTM metric to .pkl file with computation results"""
import os
import pickle
import re
import json
from absl import app
from absl import flags
from tqdm import tqdm

from run_alphafold_stage2a_comp import _read_target_file, FLAGS, MAX_MSA_DEPTH_MONO

from alphafold.data.complex import make_complex_features
from alphafold.model import config
from alphafold.common import confidence, protein
from alphafold.data.complex import initialize_template_feats

import numpy as np
# Internal import (7716).


def _get_feature_dict(target, target_name):
  """
  piece of code from run_alphafold_stage2a_comp.py
  """
  homo_copy = []
  seq_names = []
  for homo in target['split']:
    for seq_name, seq_copy in homo.items():
      homo_copy.append(int(seq_copy))
      seq_names.append(seq_name)

  # Retrieve pre-generated features of monomers (single protein sequences)
  feature_dicts = []
  for seq_name in seq_names:
    feature_dir = os.path.join(FLAGS.feature_dir, seq_name)
    if not os.path.exists(feature_dir):
      raise SystemExit("Error: ", feature_dir, "does not exists")

    # load pre-generated features as a pickled dictionary.
    features_input_path = os.path.join(feature_dir, 'features.pkl')
    with open( features_input_path, "rb" ) as f:
      mono_feature_dict = pickle.load(f)
      N = len(mono_feature_dict["msa"])
      L = len(mono_feature_dict["residue_index"])
      T = len(mono_feature_dict["template_domain_names"])
      print(f"Info: {target_name} found monomer {seq_name} msa_depth = {N}, seq_len = {L}, num_templ = {T}")
      if N > MAX_MSA_DEPTH_MONO:
          print(f"Info: {target_name} MSA size is too large, reducing to {MAX_MSA_DEPTH_MONO}")
          mono_feature_dict["msa"] = mono_feature_dict["msa"][:MAX_MSA_DEPTH_MONO,:]
          mono_feature_dict["deletion_matrix_int"] = mono_feature_dict["deletion_matrix_int"][:MAX_MSA_DEPTH_MONO,:]
          mono_feature_dict['num_alignments'][:] = MAX_MSA_DEPTH_MONO
      feature_dicts.append( mono_feature_dict )

  # Make features for complex structure prediction using monomer structures if necessary
  if len(seq_names) == 1 and homo_copy[0] == 1:   # monomer structure prediction
    feature_dict = feature_dicts[0]
    seq_len = len(feature_dict["residue_index"])
    Ls = [seq_len]
    if FLAGS.template_mode == 'none':
        new_tem = initialize_template_feats(0, seq_len)
        feature_dict.update(new_tem)
  else:  # complex structure prediction
    feature_dict, Ls = make_complex_features(
      feature_dicts, target_name, homo_copy, FLAGS.template_mode)

  return feature_dict

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # read list of target files to update with pITM metrics
  target_lst = _read_target_file( FLAGS.target_lst_path )

  for target in target_lst:
    # if complex features were not saved, rebuild them
    target_name  = target['name']
    target_name = re.sub(":", "_x", target_name)
    target_name = re.sub("/", "+", target_name)
    target_dir = os.path.join(FLAGS.output_dir, target_name)
    if not FLAGS.write_complex_features:
      feature_dict = _get_feature_dict(target, target_name)


    for pkl_file in os.listdir(target_dir):
      if ".pkl" in pkl_file and pkl_file.startswith("model_") and '_ptm_' in pkl_file:
        model_name = os.path.basename( pkl_file ).split(".")[0]
        # model_config = config.model_config(pkl_file[:7])
        # breaks = np.linspace(
        #   0., model_config.model.heads.predicted_aligned_error.max_error_bin,
        #   model_config.model.heads.predicted_aligned_error.num_bins - 1)
        pkl_path = os.path.join(target_dir, pkl_file)
        try:
            result = pickle.load(open(pkl_path, "rb"))
            #print(f"Info: {target_name} {pkl_path} loaded")
        except (EOFError,IOError) as error:
            print(f"Warning: {target_name} {error} encountered, check the pickle file")
            continue

        final_atom_mask = result['structure_module']['final_atom_mask']
        b_factors = result['plddt'][:, None] * final_atom_mask

        unrelaxed_protein = protein.from_prediction(feature_dict,
                                            result,
                                            b_factors=b_factors,
                                            remove_leading_feature_dimension=False
                                            )
        pdb_output_path = os.path.join(target_dir, model_name+'.pdb')
        with open(pdb_output_path, 'w') as f:
          f.write(protein.to_pdb(unrelaxed_protein))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'target_lst_path',
      'output_dir',
      'feature_dir',
      'template_mode',
  ])

  app.run(main)
