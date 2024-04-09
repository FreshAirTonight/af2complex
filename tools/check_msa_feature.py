# Cropping input features of a monomer, e.g., reducing the MSA size or template number
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology
#
"""AF2Complex: protein complex structure prediction with deep learning"""
import json
import os
import pickle
import random
import sys
import time
import re

from typing import Dict, Type
from fileinput import hook_compressed
#from memory_profiler import profile

parent_dir = os.path.dirname( os.path.dirname(os.path.realpath(__file__)) )
af_dir = os.path.join(parent_dir, 'src')
sys.path.append(af_dir)

from absl import app
from absl import flags
from absl import logging


#from alphafold.data.complex import read_af2c_target_file,initialize_template_feats
from alphafold.common.residue_constants import restypes


import numpy as np
# Internal import (7716).


#flags.DEFINE_string('feature_file', None, 'Path to a feature pickle file.')

flags.DEFINE_integer('num_msa_seq', 1, 'The maximum MSA sequence to print out', lower_bound=0)
flags.DEFINE_integer('num_templates', 0, 'The maximum PDB template to check their IDs', lower_bound=0)


FLAGS = flags.FLAGS
Flag = Type[FLAGS]


## number to AA mapping
aa_dict = {i: restypes[i] for i in range(20)}
aa_dict[21] = '-'


##################################################################################################


##################################################################################################
#@profile  # debugging possible memory leak with pickle load
def main(argv):

  # load pre-generated features as a pickled dictionary.
  features_input_path = argv[1]

  if not os.path.exists(features_input_path):
    raise Exception(f"Error: could not locate feature input file under {feature_dir}" )

  with hook_compressed(features_input_path, "rb") as f:
    mono_feature_dict = None
    mono_feature_dict = pickle.load(f)
    N = len(mono_feature_dict["msa"])
    L = len(mono_feature_dict["residue_index"])
    T = mono_feature_dict["template_all_atom_positions"].shape[0]
    print(f"Info: found msa_depth = {N}, seq_len = {L}, num_templ = {T}")

    msa = mono_feature_dict["msa"][:FLAGS.num_msa_seq,:]
    species = mono_feature_dict['msa_species_identifiers'][:FLAGS.num_msa_seq]
    templ_name = mono_feature_dict["template_domain_names"][:FLAGS.num_templates]

    for ind in range(msa.shape[0]):
        print(f"Info: msa = {ind} species = {species[ind].decode('utf-8')}")
        seq = ""
        for i in range(msa.shape[1]):
            seq += aa_dict[msa[ind,i]]
        print(f"Info: {seq}")

    for ind in range(templ_name.shape[0]):
        print(f"Info: template {ind} {templ_name[ind].decode('utf-8')}")






if __name__ == '__main__':
  #flags.mark_flags_as_required([
      #'target_lst_path',
      #'num_msa_seq',
      #'num_templates',
  #])

  app.run(main)
