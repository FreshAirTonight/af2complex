# To prepare features for complex structure prediction, using features prepared
# for monomers
#
# The implementation of these features used some ideas from ColabFold
# Mu Gao

import pickle
import numpy as np

from alphafold.data import pipeline

CHAIN_BREAK_GAP = 200   ### used for separate residues between monomers

##################################################################################################
def initialize_template_feats(num_templates_, num_res_):
  return {
      'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
      'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37], np.float32),
      'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
      'template_domain_names': np.empty([num_templates_], dtype=object),
      'template_sequence': np.empty([num_templates_], dtype=object),
      'template_sum_probs': np.zeros([num_templates_,1], np.float32),
  }

##################################################################################################
def _chain_break(idx_res, Ls, length=CHAIN_BREAK_GAP):
  # Minkyung's code
  # add big enough number to residue index to indicate chain breaks
  L_prev = 0
  for L_i in Ls[:-1]:
    idx_res[L_prev+L_i:] += length
    L_prev += L_i
  return idx_res
  
##################################################################################################
def make_complex_features(feature_dicts, name, copy_num_lst, template_mode):
  '''
  Make feature dict for a complex using feature dicts of monomers.
  Padding gaps to represent individual monomers in each alignment sequence/matrix
  Each original alignment now has copy_num alignments, each with copy_num * num_mono_res
  Assemble template features
  '''

  num_monomer = len(feature_dicts)
  msa_depth  = 0
  msa_length = 0
  full_sequence = ''
  Ls = []
  for n in range(num_monomer):
      features = feature_dicts[n]
      copy_num = copy_num_lst[n]

      mono_sequence = features['sequence'][0].decode()
      num_res = len(mono_sequence)
      msa  = features['msa']
      mtx  = features['deletion_matrix_int']
      num_aln = msa.shape[0]
      full_sequence += mono_sequence * copy_num
      for L in [num_res] * copy_num:
          Ls.append(L)

      msa_depth  += num_aln * copy_num
      msa_length += num_res * copy_num

  full_num_res = len(full_sequence)
  new_msa = np.zeros([msa_depth, msa_length], dtype=np.int32) + 21
  new_mtx = np.zeros([msa_depth, msa_length], dtype=np.int32)
  new_aln = np.zeros([msa_length], dtype=np.int32) + msa_depth
  #print(f"Info: oligomer MSA shape = {new_msa.shape},  number of monomers = {Ls}")

  # take care of MSA features for complex prediction
  col = 0; row = 0; full_num_tem = 0
  for n in range(num_monomer):
      features = feature_dicts[n]
      copy_num = copy_num_lst[n]
      msa  = features['msa']
      mtx  = features['deletion_matrix_int']
      num_aln = msa.shape[0]
      num_res = msa.shape[1]
      for i in range(copy_num):
        col_ = col + num_res
        row_ = row + num_aln
        new_msa[row:row_,col:col_] = msa
        new_mtx[row:row_,col:col_] = mtx
        col = col_; row = row_
        full_num_tem += len(features['template_domain_names'])

  new_feature_dict = {}
  new_feature_dict["msa"] = new_msa
  new_feature_dict["deletion_matrix_int"] = new_mtx
  new_feature_dict["num_alignments"] = new_aln

  new_feature_dict.update(pipeline.make_sequence_features(full_sequence, name, full_num_res))

  # take care of templates, assemble monomer templates for complex prediction, if necessary
  col = 0; row = 0
  if template_mode == 'oligomer':
    new_tem = initialize_template_feats(full_num_tem, full_num_res)
    for n in range(num_monomer):
      features = feature_dicts[n]
      copy_num = copy_num_lst[n]
      num_tem = len(features['template_domain_names'])
      num_res = len(features["residue_index"])
      for i in range(copy_num):
        col_ = col + num_res
        row_ = row + num_tem
        if num_tem > 0:
            new_tem['template_aatype'][row:row_,col:col_,:] = features['template_aatype']
            new_tem['template_all_atom_masks'][row:row_,col:col_,:] = features['template_all_atom_masks']
            new_tem['template_all_atom_positions'][row:row_,col:col_,...] = features['template_all_atom_positions']
            new_tem['template_domain_names'][row:row_] = features['template_domain_names']
            new_tem['template_sequence'][row:row_]  = features['template_sequence']
            new_tem['template_sum_probs'][row:row_] = features['template_sum_probs']
        col = col_; row = row_
  else:
      new_tem = initialize_template_feats(0, full_num_res)

  new_feature_dict.update(new_tem)

  resid = new_feature_dict['residue_index']
  new_feature_dict['residue_index'] = _chain_break(resid, Ls)

  return new_feature_dict, Ls

##################################################################################################
from string import ascii_uppercase,ascii_lowercase,digits
CHAIN_IDs = ascii_uppercase + ascii_lowercase + digits

def get_mono_chain(seq_names, homo_copy, Ls):
  '''
  For output purposes

  Returns dictionary with the names of individual monomer chains to model
  '''
  chains = {}
  chain_counter = 0
  num_monomer = len(seq_names)
  for i in range(num_monomer):
    mono_name = seq_names[i]
    num_copy = homo_copy[i]
    for j in range(num_copy):
      L = Ls[chain_counter]
      chain_id = CHAIN_IDs[chain_counter]
      name = mono_name + '_' + str(j+1)
      chains[chain_id] = name
      chain_counter += 1
  return chains
##################################################################################################
