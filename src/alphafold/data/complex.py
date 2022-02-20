# To prepare features for complex structure prediction, using features prepared
# for monomers
#
# The "chain_break" code was taken and modified from an early version of ColabFold
#
# Mu Gao and Davi Nakajima An

import os
import pickle
import numpy as np
from copy import deepcopy
from typing import List, Callable, Dict, Tuple

from alphafold.common import protein
from alphafold.data import pipeline, pipeline_multimer, feature_processing
from alphafold.data.msa_pairing import _correct_post_merged_feats_af2complex


CHAIN_BREAK_GAP = 200   ### used for separate residues between monomers

##################################################################################################
def initialize_template_feats(num_templates_, num_res_, is_multimer=False):
  if is_multimer:
    return {
        'template_aatype': np.zeros([num_templates_, num_res_], np.int32),
        'template_all_atom_mask': np.zeros([num_templates_, num_res_, 37], np.float32),
        'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
        'all_atom_positions': np.zeros([num_res_, 37, 3], np.float32),
        'template_domain_names': np.empty([num_templates_], dtype=object),
        'template_sequence': np.empty([num_templates_], dtype=object),
        'template_sum_probs': np.zeros([num_templates_,1], np.float32),
        'asym_id': np.zeros([num_res_], np.float32),
        'sym_id': np.zeros([num_res_], np.float32),
        'entity_id': np.zeros([num_res_], np.float32),
    }

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
  # Minkyung's code modified. Her original code applies to the residue_index feature
  # resulted from the pipeline.make_sequence_features. The modified code applies to
  # pdb residue index (0-based) of each monomer. This modification can be applied to
  # process a discontinuous domain extracted from a monomer sequence.
  #
  # To add a big enough number to residue index to indicate chain breaks
  L_prev = Ls[0]
  for L_i in Ls[1:]:
    idx_res[L_prev:L_prev+L_i] += length + idx_res[L_prev-1]
    L_prev += L_i
  return idx_res

##################################################################################################

def load_monomer_feature(target, flags):
  '''
  Retrieve pre-generated features of monomers (single protien sequences)
  '''
  monomers = target['split']
  feature_dir_base = flags.feature_dir

  # required for multimer feature preprocessing
  is_multimer_np = "multimer_np" in flags.model_preset
  all_chain_features = {}
  chain_id = 0

  for monomer in monomers:
    seq_name = monomer['mono_name']
    feature_dir = os.path.join(feature_dir_base, seq_name)

    if not os.path.exists(feature_dir):
      raise SystemExit("Error: ", feature_dir, "does not exists")

    # load pre-generated features as a pickled dictionary.
    features_input_path = os.path.join(feature_dir, 'features.pkl')
    with open(features_input_path, "rb") as f:
      mono_feature_dict = pickle.load(f)
      N = len(mono_feature_dict["msa"])
      L = len(mono_feature_dict["residue_index"])
      T = mono_feature_dict["template_all_atom_positions"].shape[0]
      print(f"Info: {target['name']} found monomer {seq_name} msa_depth = {N}, seq_len = {L}, num_templ = {T}")
      if N > flags.max_mono_msa_depth:
          print(f"Info: {seq_name} MSA size is too large, reducing to {flags.max_mono_msa_depth}")
          mono_feature_dict["msa"] = mono_feature_dict["msa"][:flags.max_mono_msa_depth,:]
          mono_feature_dict["deletion_matrix_int"] = mono_feature_dict["deletion_matrix_int"][:flags.max_mono_msa_depth,:]
          mono_feature_dict['num_alignments'][:] = flags.max_mono_msa_depth
      if T > flags.max_template_hits:
          print(f"Info: {seq_name} reducing the number of structural templates to {flags.max_template_hits}")
          mono_feature_dict["template_aatype"] = mono_feature_dict["template_aatype"][:flags.max_template_hits,...]
          mono_feature_dict["template_all_atom_masks"] = mono_feature_dict["template_all_atom_masks"][:flags.max_template_hits,...]
          mono_feature_dict["template_all_atom_positions"] = mono_feature_dict["template_all_atom_positions"][:flags.max_template_hits,...]
          mono_feature_dict["template_domain_names"] = mono_feature_dict["template_domain_names"][:flags.max_template_hits]
          mono_feature_dict["template_sequence"] = mono_feature_dict["template_sequence"][:flags.max_template_hits]
          mono_feature_dict["template_sum_probs"] = mono_feature_dict["template_sum_probs"][:flags.max_template_hits,:]

      if is_multimer_np:
          for i in range(monomer["copy_number"]):
            f_dict = pipeline_multimer.convert_monomer_features_af2complex(
                mono_feature_dict, chain_id=protein.PDB_CHAIN_IDS[chain_id]) # must update chain_id for each homo copy
            f_dict.update(monomer.copy())
            all_chain_features[protein.PDB_CHAIN_IDS[chain_id]] = f_dict
            chain_id += 1
      monomer['feature_dict'] = mono_feature_dict

    if is_multimer_np:
      all_chain_features = pipeline_multimer.add_assembly_features_af2complex(all_chain_features)
      feature_processing.process_unmerged_features(all_chain_features)
      feature_dicts_multimer = [v for k,v in all_chain_features.items()]
      monomers = feature_processing.crop_chains(
                        feature_dicts_multimer,
                        msa_crop_size=flags.msa_crop_size_mono,
                        pair_msa_sequences=False,
                        max_templates=flags.max_template_hits)
      monomer_keys = ('mono_name', 'copy_number', 'domain_range')
      for i, monomer in enumerate(monomers):
        f_dict = {k:v for k, v in monomer.items() if k not in monomer_keys}
        new_monomer = {
          'feature_dict': f_dict,
          'mono_name': monomer['mono_name'],
          'copy_number': 1, #monomer['copy_number'],
          'domain_range': monomer['domain_range']
        }
        monomers[i] = new_monomer

  return monomers

def load_multimer_feature(target, flags):
  '''
  Retrieve pre-generated features of multimers (created by AF-Multimer data pipeline)
  '''
  feature_dir_base = flags.feature_dir

  complex_name = target['name']
  feature_dir = os.path.join(feature_dir_base, complex_name)

  if not os.path.exists(feature_dir):
    raise SystemExit("Error: ", feature_dir, "does not exists")

  # load pre-generated features as a pickled dictionary.
  features_input_path = os.path.join(feature_dir, 'features.pkl')
  with open(features_input_path, "rb") as f:
    feature_dict = pickle.load(f)
    N = len(feature_dict["msa"])
    L = len(feature_dict["residue_index"])
    T = feature_dict["template_all_atom_positions"].shape[0]
    print(f"Info: {complex_name} msa_depth = {N}, seq_len = {L}, num_templ = {T}")
    if N > flags.max_mono_msa_depth:
        print(f"Info: {complex_name} MSA size is too large, reducing to {flags.max_mono_msa_depth}")
        feature_dict["msa"] = feature_dict["msa"][:flags.max_mono_msa_depth,:]
        feature_dict["deletion_matrix_int"] = feature_dict["deletion_matrix_int"][:flags.max_mono_msa_depth,:]
        feature_dict['num_alignments'][:] = flags.max_mono_msa_depth
    if T > flags.max_template_hits:
        print(f"Info: {complex_name} reducing the number of structural templates to {flags.max_template_hits}")
        feature_dict["template_aatype"] = feature_dict["template_aatype"][:flags.max_template_hits,...]
        feature_dict["template_all_atom_masks"] = feature_dict["template_all_atom_masks"][:flags.max_template_hits,...]
        feature_dict["template_all_atom_positions"] = feature_dict["template_all_atom_positions"][:flags.max_template_hits,...]
        feature_dict["template_domain_names"] = feature_dict["template_domain_names"][:flags.max_template_hits]
        feature_dict["template_sequence"] = feature_dict["template_sequence"][:flags.max_template_hits]
        feature_dict["template_sum_probs"] = feature_dict["template_sum_probs"][:flags.max_template_hits,:]

  return feature_dict

##################################################################################################

##################################################################################################
from string import ascii_uppercase,ascii_lowercase,digits
CHAIN_IDs = ascii_uppercase + ascii_lowercase + digits

def get_mono_chain(monomers, Ls):
  '''
  For output purposes

  Returns dictionary with the names of individual monomer chains to model
  '''
  chains = {}
  chain_counter = 0

  for mono_entry in monomers:
    mono_name = mono_entry['mono_name']
    dom_range = mono_entry['domain_range']
    copy_num  = mono_entry['copy_number']
    for j in range(copy_num):
      L = Ls[chain_counter]
      chain_id = CHAIN_IDs[chain_counter]
      name = mono_name + '_' + str(j+1)
      if dom_range is not None:
          name = mono_name + '|' + ','.join(dom_range) + '|' + str(j+1)
      chains[chain_id] = name
      chain_counter += 1
  return chains
##################################################################################################

def make_complex_features(target, flags):
  if flags.model_preset == "multimer_np":
    return proc_mono_feats_for_af2mult(target, flags)
  elif flags.model_preset in ("monomer_ptm", "monomer"):
    return proc_mono_feats_for_af2mono(target, flags)
  elif flags.model_preset == "multimer":
    return proc_passthrough_for_orig_af(target, flags)
  else:
    raise "feature processing for model_preset not configured"

def run_feature_pipeline(
  steps: List[Callable],
  curr_input: Dict,
) -> Tuple[Dict, List, List[int]]:
  """Runs a sequence of feature preprocessing methods

  Args:
    steps: a list of methods that modify the current feature dictionary

  Returns:
    new_feature_dict: the final feature_dict to be passed to the model
    mono_chains:  dictionary with the names of individual monomer chains
      to model
    Ls: the length of each chain being modeled
  """

  for func in steps:
    curr_input = func(curr_input)

  new_feature_dict = curr_input['new_feature_dict']
  monomers = curr_input['monomers']
  Ls = curr_input['Ls']
  # get final chain IDs for output
  mono_chains = get_mono_chain(monomers, Ls)

  return new_feature_dict, mono_chains, Ls

def proc_passthrough_for_orig_af(target: Dict, flags):
  """Loads features created by AF-Multimer or AF2 pipeline for a target and
  returns it.

  Args:
    target: dictionary with the items:
        name: name of the multimer,
        split: information about each monomer composing the multimer,
        full: a string denoting all stoichiometry and domains of all monomers
          composing the multimer to be modeled,
    flags: variable containing inference configuration

  Returns:
    new_feature_dict: the final feature_dict to be passed to the model
    mono_chains:  dictionary with the names of individual monomer chains
    Ls: the length of each monomer chain composing the multimer
  """
  feat_dict = load_multimer_feature(target, flags)

  chains = {}
  chain_counter = 0
  Ls = []
  num_monomers = int(max(feat_dict['entity_id']))
  for monomer_index in range(1, num_monomers+1):
    entity_mask = feat_dict['entity_id'] == monomer_index
    # get length of monomer
    L = (feat_dict['sym_id'][entity_mask] == 1).sum()
    Ls.append(int(L))
    # get copy number of monomer
    copy_num = int((feat_dict['sym_id'][entity_mask]).max())
    # split order must match feature creation pipeline
    mono_name = target['split'][monomer_index-1]['mono_name']
    dom_range = target['split'][monomer_index-1]['domain_range']

    for j in range(copy_num):
      chain_id = CHAIN_IDs[chain_counter]
      name = mono_name + '_' + str(j+1)
      if dom_range is not None:
          name = mono_name + '|' + ','.join(dom_range) + '|' + str(j+1)
      chains[chain_id] = name
      chain_counter += 1

  return feat_dict, chains, Ls

def proc_mono_feats_for_af2mono(target: Dict, flags):
  """Defines the sequence of preprocessing steps for the
    monomer_ptm and monomer presets (i.e. using original AF2 model weights
      with predicted TM-score)

  Args:
    target: dictionary with the items:
        name: name of the multimer,
        split: information about each monomer composing the multimer,
        full: a string denoting all stoichiometry and domains of all monomers
          composing the multimer to be modeled,
    flags: variable containing inference configuration

  Returns:
    new_feature_dict: the final feature_dict to be passed to the model
    mono_chains:  dictionary with the names of individual monomer chains
    Ls: the length of each monomer chain composing the multimer
  """
  monomers = load_monomer_feature(target, flags)
  curr_input = {'monomers': monomers, 'target': target, 'flags': flags}

  return run_feature_pipeline(
    [
      targeted_domain_cropping_mono,
      join_seqs_feats_unpaired_mono,
      template_cropping_and_joining_mono,
      apply_residue_jumps_between_chains,
    ],
    curr_input
  )

def proc_mono_feats_for_af2mult(target: Dict, flags):
  """Defines the sequence of preprocessing steps for the
    multimer_np preset (i.e. using AF-Multimer model weights on
      joined monomer MSAs w/o pairing)

  Args:
    target: dictionary with the items:
        name: name of the multimer,
        split: information about each monomer composing the multimer,
        full: a string denoting all stoichiometry and domains of all monomers
          composing the multimer to be modeled,
    flags: variable containing configuration of inference run

  Returns:
    new_feature_dict: the final feature_dict to be passed to the model
    mono_chains:  dictionary with the names of individual monomer chains
    Ls: the length of each monomer chain composing the multimer
  """
  monomers = load_monomer_feature(target, flags)

  for i in range(len(monomers)):
    monomers[i]['feature_dict']['deletion_matrix_int'] = monomers[i]['feature_dict']['deletion_matrix']
  curr_input = {'monomers': monomers, 'target': target, 'flags': flags}

  def fix_aatype(curr_input):
    aatype = curr_input['new_feature_dict']['aatype']
    curr_input['new_feature_dict']['aatype'] = np.argmax(
      aatype, axis=-1).astype(np.int32)
    return curr_input

  def final_mult_processing(curr_input):
    new_feature_dict = curr_input['new_feature_dict']
    monomers = curr_input['monomers']
    new_feature_dict['deletion_matrix'] = new_feature_dict['deletion_matrix_int']

    feature_dict = feature_processing.process_final_af2complex(new_feature_dict)
    feature_dict = _correct_post_merged_feats_af2complex(feature_dict, monomers)
    num_seq = feature_dict['msa'].shape[0]
    if num_seq < 512:
      feature_dict = pipeline_multimer.pad_msa(feature_dict, 512)

    curr_input['new_feature_dict'] = feature_dict
    return curr_input

  return run_feature_pipeline(
    [
      targeted_domain_cropping_mono,
      join_seqs_feats_unpaired_mono,
      fix_aatype,
      template_cropping_and_joining_mult,
      apply_residue_jumps_between_chains,
      final_mult_processing,
    ],
    curr_input
  )

def apply_residue_jumps_between_chains(curr_input):
  """Applies a chain break between different chains by altering
    the residue_index feature

  Args:
    curr_input: a dictionary containing all the information necessary for the
      feature processing pipeline. Each method in the pipeline uses variables from this
      dictionary and updates it according to the necessary variables in the the future
      methods in the pipeline.
  """
  Ls = curr_input['Ls']
  new_feature_dict = curr_input['new_feature_dict']
  del new_feature_dict['residue_index']
  resid = deepcopy(new_feature_dict["pdb_residue_index"])
  # apply chain_break
  new_feature_dict['residue_index'] = _chain_break(resid, Ls)
  curr_input.update({'new_feature_dict': new_feature_dict})
  return curr_input

def extract_domain_mono(mono_entry):
  """Extracts the features of a domain specified by a list of domain ranges
    from a larger chain.

  Args:
    mono_entry: dictionary containing the features of a monomer in the multimer
  """
  dom_range = mono_entry['domain_range']
  mono_name = mono_entry['mono_name']
  features  = mono_entry['feature_dict']
  mono_sequence = features['sequence'][0].decode()

  msa = features['msa']
  mtx = features['deletion_matrix_int']
  resid = features['residue_index']
  if dom_range is not None:
    mono_seq = ''
    print(f"Info: extracting resid range {dom_range} of {mono_name}")
    for idx, boundary in enumerate(dom_range):
        sta,end = boundary.split('-')
        sta = int(sta) - 1
        end = int(end)
        mono_seq += mono_sequence[sta:end]
        if idx == 0:
            mono_msa = msa[:,sta:end]
            mono_mtx = mtx[:,sta:end]
            mono_res = resid[sta:end]
        else:
            mono_msa = np.concatenate((mono_msa, msa[:,sta:end]), axis=1)
            mono_mtx = np.concatenate((mono_mtx, mtx[:,sta:end]), axis=1)
            mono_res = np.concatenate((mono_res, resid[sta:end]), axis=0)
    all_gap_rows = ~np.all(mono_msa == 21, axis=1)   ## remove ones with gaps only
    features['msa']= mono_msa[all_gap_rows]
    features['deletion_matrix_int'] = mono_mtx[all_gap_rows]
    mono_entry['feature_dict'] = features
    mono_sequence = mono_seq
    features['residue_index'] = mono_res

  return mono_entry, mono_sequence

def extract_template_domain_mono(mono_entry):
  """Extracts the template features of a domain specified by a list of domain ranges
    from a larger chain.

  Args:
    mono_entry: dictionary containing the features of a monomer in the multimer
  """
  features  = mono_entry['feature_dict']
  copy_num  = mono_entry['copy_number']
  dom_range = mono_entry['domain_range']
  mono_sequence = features['sequence'][0].decode()
  if dom_range is not None:
    mono_seq = ''
    for idx, boundary in enumerate(dom_range):
        sta,end = boundary.split('-')
        sta = int(sta) - 1
        end = int(end)
        mono_seq += mono_sequence[sta:end]
        if idx == 0:
            temp_aatype = features['template_aatype'][:,sta:end,:]
            temp_all_atom_masks = features['template_all_atom_masks'][:,sta:end,:]
            temp_all_atom_positions = features['template_all_atom_positions'][:,sta:end,...]
        else:
            temp_aatype = np.concatenate((temp_aatype,
                features['template_aatype'][:,sta:end,:]), axis=1)
            temp_all_atom_masks = np.concatenate((temp_all_atom_masks,
                features['template_all_atom_masks'][:,sta:end,:]), axis=1)
            temp_all_atom_positions = np.concatenate((temp_all_atom_positions,
                features['template_all_atom_positions'][:,sta:end,...]), axis=1)
    features['template_aatype'] = temp_aatype
    features['template_all_atom_masks'] = temp_all_atom_masks
    features['template_all_atom_positions'] = temp_all_atom_positions

  return features

def extract_template_domain_mult(mono_entry):
  """Extracts the template features (in the format of AF-Multimer) of a domain specified
    by a list of domain ranges from a larger chain.

  Args:
    mono_entry: dictionary containing the features of a monomer in the multimer
  """
  features  = mono_entry['feature_dict']
  copy_num  = mono_entry['copy_number']
  dom_range = mono_entry['domain_range']
  mono_sequence = features['sequence'][0].decode()
  if dom_range is not None:
    mono_seq = ''
    for idx, boundary in enumerate(dom_range):
        sta,end = boundary.split('-')
        sta = int(sta) - 1
        end = int(end)
        mono_seq += mono_sequence[sta:end]
        if idx == 0:
            temp_aatype = features['template_aatype'][:,sta:end]
            temp_all_atom_masks = features['template_all_atom_mask'][:,sta:end,:]
            temp_asym_ids = features['asym_id'][sta:end]
            temp_sym_ids = features['sym_id'][sta:end]
            temp_entity_ids = features['entity_id'][sta:end]
            temp_all_atom_positions = features['template_all_atom_positions'][:,sta:end,...]
        else:
            temp_aatype = np.concatenate((temp_aatype,
                features['template_aatype'][:,sta:end]), axis=1)
            temp_all_atom_masks = np.concatenate((temp_all_atom_masks,
                features['template_all_atom_mask'][:,sta:end,:]), axis=1)
            temp_asym_ids = np.concatenate((temp_asym_ids,
                features['asym_id'][sta:end]), axis=0)
            temp_sym_ids = np.concatenate((temp_sym_ids,
                features['sym_id'][sta:end]), axis=0)
            temp_entity_ids= np.concatenate((temp_entity_ids,
                features['entity_id'][sta:end]), axis=0)
            temp_all_atom_positions = np.concatenate((temp_all_atom_positions,
                features['template_all_atom_positions'][:,sta:end,...]), axis=1)
    features['template_aatype'] = temp_aatype
    features['template_all_atom_mask'] = temp_all_atom_masks
    features['asym_id'] = temp_asym_ids
    features['sym_id'] = temp_sym_ids
    features['entity_id'] = temp_entity_ids
    features['template_all_atom_positions'] = temp_all_atom_positions

  return features

def targeted_domain_cropping_mono(curr_input):
  """Extracts the domain of a monomer if necessary.

  Also updates curr_input with:
      msa_depth, msa_length, Ls, pdb_res_id, pdb_cha_id, and full_sequence

  Args:
    curr_input: a dictionary containing all the information necessary for the
      feature processing pipeline. Each method in the pipeline uses variables from this
      dictionary and updates it according to the necessary variables in the the future
      methods in the pipeline.
  """
  # first run through monomers of a target to crop each monomer if necessary
  monomers = curr_input['monomers']

  msa_depth  = 0; msa_length = 0; chain_id = 0
  full_sequence = ''
  Ls = []   ## length of each monomer chain
  pdb_res_id = None
  pdb_cha_id = None

  for mono_entry in monomers:
      # extract domain from monomer if necessary
      mono_entry, mono_sequence = extract_domain_mono(mono_entry)

      mono_name = mono_entry['mono_name']
      features  = mono_entry['feature_dict']
      copy_num  = mono_entry['copy_number']
      msa   = features['msa']
      mtx   = features['deletion_matrix_int']
      resid = features['residue_index']

      num_aln = msa.shape[0]
      full_sequence += mono_sequence * copy_num
      num_res = len(mono_sequence)

      for L in [num_res] * copy_num:
          Ls.append(L)
          if pdb_res_id is None:
            pdb_res_id = resid
            pdb_cha_id = np.zeros([num_res])
          else:
            pdb_res_id = np.concatenate((pdb_res_id, resid))
            pdb_cha_id = np.concatenate((pdb_cha_id, np.zeros([num_res])+chain_id))
          chain_id += 1

      msa_depth  += num_aln * copy_num
      msa_length += num_res * copy_num

  curr_input.update(
    {
      'msa_depth': msa_depth,
      'msa_length': msa_length,
      'Ls': Ls,
      'pdb_res_id': pdb_res_id,
      'pdb_cha_id': pdb_cha_id,
      'full_sequence': full_sequence
    }
  )
  return curr_input


def join_seqs_feats_unpaired_mono(curr_input):
  """Joins the features of each monomer chain containing unpaired MSAs.

  Also updates curr_input with:
      full_num_tem, full_num_res, and new_feature_dict

  Args:
    curr_input: a dictionary containing all the information necessary for the
      feature processing pipeline. Each method in the pipeline uses variables from this
      dictionary and updates it according to the necessary variables in the the future
      methods in the pipeline.
  """
  monomers = curr_input['monomers']
  msa_length = curr_input['msa_length']
  msa_depth = curr_input['msa_depth']
  pdb_res_id = curr_input['pdb_res_id']
  pdb_cha_id = curr_input['pdb_cha_id']
  full_sequence = curr_input['full_sequence']
  target = curr_input['target']

  ### initialize complex feature
  full_num_res = len(full_sequence)
  new_msa = np.zeros([msa_depth, msa_length], dtype=np.int32) + 21
  new_mtx = np.zeros([msa_depth, msa_length], dtype=np.int32)
  new_aln = np.zeros([msa_length], dtype=np.int32) + msa_depth

  # take care of MSA features for complex prediction
  col = 0; row = 0; full_num_tem = 0
  for mono_entry in monomers:
      features = mono_entry['feature_dict']
      copy_num = mono_entry['copy_number']

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
  new_feature_dict["num_alignments"]      = new_aln
  new_feature_dict["pdb_residue_index"]   = pdb_res_id.astype(np.int32)
  new_feature_dict["pdb_chain_index"]     = pdb_cha_id.astype(np.int32)
  new_feature_dict.update(pipeline.make_sequence_features(full_sequence, target['name'], full_num_res))

  curr_input.update({
    'full_num_tem': full_num_tem,
    'full_num_res': full_num_res,
    'new_feature_dict': new_feature_dict,
    })
  return curr_input

def template_cropping_and_joining_mono(curr_input):
  """Extracts domains from each monomer's template features and joins the
    monomer's template features.

    Also updates curr_input with:
      new_feature_dict (updated)

  Args:
    curr_input: a dictionary containing all the information necessary for the
      feature processing pipeline. Each method in the pipeline uses variables from this
      dictionary and updates it according to the necessary variables in the the future
      methods in the pipeline.
  """
  flags = curr_input['flags']
  full_num_res = curr_input['full_num_res']
  full_num_tem = curr_input['full_num_tem']
  monomers = curr_input['monomers']
  new_feature_dict = curr_input['new_feature_dict']

  col = 0; row = 0
  if not flags.no_template:
    new_tem = initialize_template_feats(full_num_tem, full_num_res, False)
    for mono_entry in monomers:
      features = extract_template_domain_mono(mono_entry)

      copy_num = mono_entry['copy_number']
      num_res = features['template_aatype'].shape[1]
      num_tem = len(features['template_domain_names'])
      for i in range(copy_num):
        col_ = col + num_res
        row_ = row + num_tem
        if num_tem > 0:
          new_tem['template_all_atom_positions'][row:row_,col:col_,...] = features['template_all_atom_positions']
          new_tem['template_domain_names'][row:row_] = features['template_domain_names']
          new_tem['template_sequence'][row:row_]  = features['template_sequence']
          new_tem['template_sum_probs'][row:row_] = features['template_sum_probs']
          new_tem['template_aatype'][row:row_,col:col_,:] = features['template_aatype']
          new_tem['template_all_atom_masks'][row:row_,col:col_,:] = features['template_all_atom_masks']
        col = col_; row = row_
    new_feature_dict.update(new_tem)

  curr_input.update({'new_feature_dict': new_feature_dict})
  return curr_input

def template_cropping_and_joining_mult(curr_input):
  """Extracts domains from each monomer's template features (in the format of AF-Multimer)
    and joins the monomer's template features (in the format of AF-Multimer).

    Also updates curr_input with:
      new_feature_dict (updated)

  Args:
    curr_input: a dictionary containing all the information necessary for the
      feature processing pipeline. Each method in the pipeline uses variables from this
      dictionary and updates it according to the necessary variables in the the future
      methods in the pipeline.
  """

  flags = curr_input['flags']
  full_num_res = curr_input['full_num_res']
  full_num_tem = curr_input['full_num_tem']
  monomers = curr_input['monomers']
  new_feature_dict = curr_input['new_feature_dict']

  col = 0; row = 0
  if not flags.no_template:
    new_tem = initialize_template_feats(full_num_tem, full_num_res, True)
    for mono_entry in monomers:
      features = extract_template_domain_mult(mono_entry)
      copy_num = mono_entry['copy_number']

      num_res = features['template_aatype'].shape[1]
      num_tem = len(features['template_domain_names'])
      for i in range(copy_num):
        col_ = col + num_res
        row_ = row + num_tem
        if num_tem > 0:
          new_tem['template_all_atom_positions'][row:row_,col:col_,...] = features['template_all_atom_positions']
          new_tem['template_domain_names'][row:row_] = features['template_domain_names']
          new_tem['template_sequence'][row:row_]  = features['template_sequence']
          new_tem['template_sum_probs'][row:row_] = features['template_sum_probs']
          new_tem['template_all_atom_mask'][row:row_,col:col_,:] = features['template_all_atom_mask']
          new_tem['template_aatype'][row:row_,col:col_] = features['template_aatype']
          new_tem['asym_id'][col:col_] = features['asym_id']
          new_tem['sym_id'][col:col_] = features['sym_id']
          new_tem['entity_id'][col:col_] = features['entity_id']

        col = col_; row = row_
    new_feature_dict.update(new_tem)

  curr_input.update({'new_feature_dict': new_feature_dict})
  return curr_input
