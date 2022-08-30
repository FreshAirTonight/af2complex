# To prepare features for complex structure prediction, using features prepared
# for monomers
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology, 2021-2022
#
import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
import re
import fileinput
from copy import deepcopy
from typing import List, Callable, Dict, Tuple

from alphafold.common import protein
from alphafold.data import pipeline, pipeline_multimer, feature_processing
from alphafold.data.msa_pairing import _correct_post_merged_feats_af2complex
from alphafold.data import msa_pairing


CHAIN_BREAK_GAP = 200   ### used for separate residues between monomers



##################################################################################################
def _process_target_name( target_name: str ):
    target_name = re.sub(':', "_x", target_name)
    target_name = re.sub('/', "+",  target_name)
    target_name = re.sub('\|', "_", target_name)
    return target_name

# read either a single target string or a input list of targets in a file
# each line has a format like: <target> <length> (output_name), the output_name is optional.
# In <target>, use monomer:num to indicate num of copies in a homooligomer
# and name1/name2 to indicate heterooligomer. For example, TarA:2/TarB is 2 copies of TarA and 1 TarB
# Use TarA|20-100;120-160:2/TarB to model domains defined by residue id range [20-100;120-160] of TarA
def read_af2c_target_file( data_lst_file: str ):
    target_lst = []
    if not os.path.exists( data_lst_file ):  ### input is a single target in strings
        fields = data_lst_file.split(',')
        fullname = name = fields[0]
        if len(fields) >= 2:
            name = fields[1]
        else:
            name = _process_target_name(name)

        if len(fields) >= 3:
            adj_list_path = fields[2]
        else:
            adj_list_path = None

        model = None
        model_preset = None
        msa_pairing = None
        target_lst.append( {
            'full':fullname, 'name':name, 'adj_list_path': adj_list_path,
            'model': None, 'model_preset': None, 'msa_pairing': None,
            } )

    else: ### input are a list of targets in a file
      with open( data_lst_file ) as file:
        model = None
        model_preset = None
        msa_pairing = None

        for line in file:
          if len(line.strip()) == 0:
            continue

          if line.startswith("*"):
            line = line[1:].strip() # strip "\n"
            fields = line.split()
            if len(fields) >= 1:
              model = fields[0].split(',')
            if len(fields) >= 2:
              model_preset = fields[1]
            if len(fields) >= 3:
              msa_pairing = fields[2]
            continue

          if line.startswith("#"):
              continue
          line = line.strip() # strip "\n"
          fields = line.split()
          fullname = name = fields[0]
          if len(fields) > 2 and not fields[2].startswith("#"):
              name = fields[2]
          else:
              name = _process_target_name(name)
          if len(fields) == 4 and not fields[3].startswith("#"):
              adj_list_path = fields[3]
          else:
              adj_list_path = None
          target_lst.append( {
            'full':fullname, 'name':name, 'adj_list_path': adj_list_path,
            'model': model, 'model_preset': model_preset, 'msa_pairing': msa_pairing } )

    # process the components of a complex target if detected
    for target in target_lst:
        complex = target['full']
        monomers = []
        asym_id_list = []
        chain_id = 0
        submetric = complex.split('+')
        for metric_group in submetric:
          asym_id_list.append([])
          subfields = metric_group.split('/')
          for item in subfields:
              cols = item.split(':')
              subcols = cols[0].split('|')
              mono_name = subcols[0]
              if len(subcols) > 1:
                  domain_range = subcols[1].split(';')
              else:
                  domain_range = None
              if len(cols) == 1:
                  monomers.append( {'mono_name': mono_name, 'copy_number':1,
                              'domain_range':domain_range} )  ### monomer
                  asym_id_list[-1].append(chain_id)
                  chain_id += 1
              elif len(cols) > 1:
                  monomers.append( {'mono_name': mono_name, 'copy_number':int(cols[1]),
                              'domain_range':domain_range} )
                  for i in range(int(cols[1])):
                    asym_id_list[-1].append(chain_id)
                    chain_id += 1
        if len(submetric) == 1:
          asym_id_list = None
        if len(monomers) >= 1:
            target['split'] = monomers
        target['asym_id_list'] = asym_id_list

    return target_lst
##################################################################################################

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
      'template_domain_names': np.empty([num_templates_], dtype=str),
      'template_sequence': np.empty([num_templates_], dtype=str),
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

def load_monomer_feature(target: Dict, flags):
  '''
  Retrieve pre-generated features of monomers (single protien sequences)
  '''
  monomers = target['split']
  feature_dir_base = flags.feature_dir

  # required for multimer feature preprocessing
  is_multimer_np = "multimer_np" in flags.model_preset
  is_multimer_np = is_multimer_np or ("monomer_ptm" in flags.model_preset and flags.msa_pairing in ("all", "cyclic", "linear", "custom"))

  all_chain_features = {}
  chain_id = 0

  all_mono_saved_feats = {}
  for monomer in monomers:
    seq_name = monomer['mono_name']
    feature_dir = os.path.join(feature_dir_base, seq_name)

    if not os.path.exists(feature_dir):
      raise SystemExit("Error: ", feature_dir, "does not exists")

    # load pre-generated features as a pickled dictionary.
    features_input_path = os.path.join(feature_dir, 'features.pkl.gz')
    if not os.path.exists(features_input_path):
      features_input_path = os.path.join(feature_dir, 'features.pkl')
      if not os.path.exists(features_input_path):
          raise Exception(f"Error: {seq_name} could not locate feature input file under {feature_dir}" )

    with fileinput.hook_compressed(features_input_path, "rb") as f:
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
          if 'msa_species_identifiers' in mono_feature_dict:
            mono_feature_dict['msa_species_identifiers'] = mono_feature_dict['msa_species_identifiers'][:flags.max_mono_msa_depth]

      if T > flags.max_template_hits:
          print(f"Info: {seq_name} reducing the number of structural templates to {flags.max_template_hits}")
          mono_feature_dict["template_aatype"] = mono_feature_dict["template_aatype"][:flags.max_template_hits,...]
          mono_feature_dict["template_all_atom_masks"] = mono_feature_dict["template_all_atom_masks"][:flags.max_template_hits,...]
          mono_feature_dict["template_all_atom_positions"] = mono_feature_dict["template_all_atom_positions"][:flags.max_template_hits,...]
          mono_feature_dict["template_domain_names"] = mono_feature_dict["template_domain_names"][:flags.max_template_hits]
          mono_feature_dict["template_sequence"] = mono_feature_dict["template_sequence"][:flags.max_template_hits]
          mono_feature_dict["template_sum_probs"] = mono_feature_dict["template_sum_probs"][:flags.max_template_hits,:]

      if T == 0 or flags.no_template:  # deal with senario no template found, or set it to a null template if requested
          mono_template_features = initialize_template_feats(1, L, is_multimer=False)
          mono_feature_dict.update(mono_template_features)

      if is_multimer_np:
          for i in range(monomer["copy_number"]):
            if flags.model_preset != 'monomer_ptm':
              f_dict = pipeline_multimer.convert_monomer_features_af2complex(
                  mono_feature_dict, chain_id=protein.PDB_CHAIN_IDS[chain_id]) # must update chain_id for each homo copy
            else:
              f_dict, saved_feats = pipeline_multimer.convert_monomer_features_af2complex(
                  mono_feature_dict, chain_id=protein.PDB_CHAIN_IDS[chain_id], save_feats_for_mono=True)
              all_mono_saved_feats[monomer['mono_name']] = saved_feats
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
                        msa_crop_size=flags.mono_msa_crop_size,
                        pair_msa_sequences=False,
                        max_templates=flags.max_template_hits)
      monomer_keys = ('mono_name', 'copy_number', 'domain_range')
      for i, monomer in enumerate(monomers):
        f_dict = {k:v for k, v in monomer.items() if k not in monomer_keys}
        if monomer['mono_name'] in all_mono_saved_feats:
          f_dict.update(all_mono_saved_feats[monomer['mono_name']])
        new_monomer = {
          'feature_dict': f_dict,
          'mono_name': monomer['mono_name'],
          'copy_number': 1, #monomer['copy_number'],
          'domain_range': monomer['domain_range']
        }
        monomers[i] = new_monomer

  return monomers

################################################################################
def load_multimer_feature(target: Dict, flags):
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

  return feature_dict
################################################################################

################################################################################
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
################################################################################

################################################################################
def make_complex_features(target: Dict, flags):
  if flags.model_preset == "multimer_np":
    return proc_mono_feats_for_af2mult(target, flags)
  elif flags.model_preset in ("monomer_ptm", "monomer"):
    return proc_mono_feats_for_af2mono(target, flags)
  elif flags.model_preset == "multimer":
    return proc_passthrough_for_orig_af(target, flags)
  else:
    raise "feature processing for model_preset not configured"

################################################################################
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

  if not np.any(new_feature_dict['asym_id'] == 0):
    new_feature_dict['asym_id'] -= 1 # make indexing start at 0

  return new_feature_dict, mono_chains, Ls
################################################################################
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
  if not np.any(feat_dict['asym_id'] == 0):
    feat_dict['asym_id'] -= 1 # make indexing start at 0

  return feat_dict, chains, Ls

################################################################################
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
  if flags.msa_pairing is not None:
    for i in range(len(monomers)):
      monomers[i]['feature_dict']['deletion_matrix_int'] = monomers[i]['feature_dict']['deletion_matrix']
  curr_input = {'monomers': monomers, 'target': target, 'flags': flags}

  def final_mono_processing(curr_input):
    curr_input['new_feature_dict']['asym_id'] = np.array(curr_input['asym_id_mono_ptm'])
    return curr_input

  return run_feature_pipeline(
    [
      targeted_domain_cropping_mono,
      add_asym_id_monomer_ptm,
      join_seqs_feats_unpaired_mono,
      template_cropping_and_joining_mono,
      apply_residue_jumps_between_chains,
      final_mono_processing,
    ],
    curr_input
  )

################################################################################
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
    if flags.msa_pairing is None:
      feature_dict = _correct_post_merged_feats_af2complex(feature_dict, monomers, False)
    else:
      feature_dict = _correct_post_merged_feats_af2complex(feature_dict, monomers, True)

    feature_dict['pdb_residue_index'] = new_feature_dict['pdb_residue_index']
    #feature_dict['pdb_chain_index'] = new_feature_dict['pdb_chain_index']
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

################################################################################
def apply_residue_jumps_between_chains(curr_input: Dict):
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

def extract_domain_mono(mono_entry: Dict):
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
    not_all_gap_rows = ~np.all(mono_msa == 21, axis=1)   ## remove ones with gaps only
    features['msa']= mono_msa[not_all_gap_rows]
    features['deletion_matrix_int'] = mono_mtx[not_all_gap_rows]
    if 'msa_species_identifiers' in features:
      features['msa_species_identifiers'] = features['msa_species_identifiers'][not_all_gap_rows]
    mono_entry['feature_dict'] = features
    mono_sequence = mono_seq
    features['residue_index'] = mono_res

  return mono_entry, mono_sequence

################################################################################
def extract_template_domain_mono(mono_entry: Dict):
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

################################################################################
def extract_template_domain_mult(mono_entry: Dict):
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

################################################################################
def targeted_domain_cropping_mono(curr_input: Dict):
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

################################################################################
def add_asym_id_monomer_ptm(curr_input: Dict):
  monomers = curr_input['monomers']
  length = curr_input['msa_length']

  asym_id = np.zeros((length,))
  chain_id = 0
  first_idx = 0
  for mono_entry in monomers:
    num_res = mono_entry['feature_dict']['msa'].shape[1]
    copy_num = mono_entry['copy_number']
    for i in range(copy_num):
      asym_id[first_idx:first_idx + num_res] = chain_id
      first_idx += num_res
      chain_id += 1

  curr_input.update({'asym_id_mono_ptm': asym_id})
  return curr_input

################################################################################
def join_seqs_feats_unpaired_mono(curr_input: Dict):
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
  flags = curr_input['flags']

  ### initialize complex feature
  full_num_res = len(full_sequence)

  full_num_tem = 0
  for mono_entry in monomers:
      features = mono_entry['feature_dict']
      copy_num = mono_entry['copy_number']
      for i in range(copy_num):
        full_num_tem += len(features['template_domain_names'])
  # take care of MSA features for complex prediction
  if flags.msa_pairing is None:
    new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_none(monomers, msa_depth, msa_length)
  elif flags.msa_pairing == 'all':
    new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_custom(monomers, flags.max_template_hits)
    if flags.model_preset == 'monomer_ptm':
      new_aln = np.zeros((new_msa.shape[1], 1), dtype=np.int32) + new_aln
  elif flags.msa_pairing == 'cyclic':
    adj_list = make_cyclic_adj_list(len(monomers))
    subgraphs = find_subgraphs_without_overlap(adj_list)
    new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_custom(
      monomers, flags.max_template_hits, subgraphs, pair_all=False)
    if flags.model_preset == 'monomer_ptm':
      new_aln = np.zeros((new_msa.shape[1], 1), dtype=np.int32) + new_aln
  elif flags.msa_pairing == 'linear':
    adj_list = make_linear_adj_list(len(monomers))
    subgraphs = find_subgraphs_without_overlap(adj_list)
    new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_custom(
      monomers, flags.max_template_hits, subgraphs, pair_all=False)
    if flags.model_preset == 'monomer_ptm':
      new_aln = np.zeros((new_msa.shape[1], 1), dtype=np.int32) + new_aln
  elif flags.msa_pairing == 'custom':
    if target['adj_list_path'] is None:
      print(f"INFO: MSA pairing mode=custom but did not define adjacency list file for target {target['name']}")
      print(f"INFO: Pairing MSA as if pairing mode=all")
      new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_custom(
        monomers, flags.max_template_hits, pair_all=True)
    else:
      adj_list = load_adjacency_list(target['adj_list_path'], len(monomers))
      print(f"INFO: using adjacency list from {target['adj_list_path']}")
      subgraphs = find_subgraphs_without_overlap(adj_list)
      new_msa, new_bert_mask, new_mtx, new_aln = msa_pairing_custom(
        monomers, flags.max_template_hits, subgraphs, pair_all=False)
    if flags.model_preset == 'monomer_ptm':
      new_aln = np.zeros((new_msa.shape[1], 1), dtype=np.int32) + new_aln
  else:
    raise f'Not yet implemented for {flags.msa_pairing}'


  new_feature_dict = {}
  new_feature_dict["msa"]                 = new_msa
  new_feature_dict["bert_mask"]           = new_bert_mask
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

################################################################################
def make_cyclic_adj_list(num_nodes: int):
  """ Creates an adjacency list for the target that corresponds to a cyclic
    protein
  """
  adj_list = {}
  for i in range(num_nodes):
    adj_list[i] = set([(i-1)% num_nodes, (i+1)%num_nodes])
  return adj_list

def make_linear_adj_list(num_nodes: int):
  """ Creates an adjacency list for the target that corresponds to a linear
    protein
  """
  adj_list = make_cyclic_adj_list(num_nodes)
  adj_list[0].remove(num_nodes - 1)
  adj_list[num_nodes-1].remove(0)
  return adj_list

def load_adjacency_list(file_path: str, num_nodes: int):
  """ Loads a target's adjacency list from a text file
  """
  adj_list = {}
  with open(file_path, 'r') as f:
      lines= f.readlines()
      f.close()

  name2idx = {}
  def remove_comment(s):
      comment_start = s.find("#")
      if comment_start == -1:
          return s
      return s[:comment_start]

  for i, line in enumerate(lines):
      if line.startswith('*'):
          line = remove_comment(line)
          line = line[1:].split('=')
          name = line[0].strip()
          idx = int(line[1].strip())
          if name in name2idx:
              raise ValueError(f"Reassigning name {name} in line {i}")

          name2idx[name] = idx

  for i in range(num_nodes):
      adj_list[i] = set()

  add_edge = lambda neighbors, n_neighbor: neighbors.union(set([n_neighbor]))
  for p in lines:
      if p.startswith('*') or p.startswith('#') or len(p.strip()) == 0:
          continue
      p = remove_comment(p)

      p = p.strip().split('-')

      a = p[0].strip()
      b = p[1].strip()
      if a in name2idx:
          a = name2idx[a]
      else:
          a = int(a)
      if b in name2idx:
          b = name2idx[b]
      else:
          b = int(b)
      adj_list[a] = add_edge(adj_list[a], b)
      adj_list[b] = add_edge(adj_list[b], a)

  return adj_list

def find_subgraphs_without_overlap(adj_list: Dict[int, Tuple[int]]):
  """ Given an adjacency list of the chains in the target, builds a
    connectivity graph and finds all the cliques in the graph that are
    non-overlapping
  """
  h_cliques = []
  max_cliques_avail = True
  G = nx.Graph()

  for i in adj_list.keys():
    for j in adj_list[i]:
      G.add_edge(i, j)

  while max_cliques_avail:
    cliques = list(nx.find_cliques(G))
    cliques = [x for x in cliques if len(x) >1]

    if len(cliques) == 0:
      max_cliques_avail = False
      continue
    to_include = cliques[0]
    h_cliques.append(to_include)
    for i in range(len(to_include)):
      for j in range(i + 1, len(to_include)):
        G.remove_edge(to_include[i], to_include[j])
  return h_cliques

def msa_pairing_none(
 monomers: Dict[str, np.ndarray], msa_depth: int, msa_length: int):
  """ Creates MSA features with no pairing among different chains.
  """
  col = 0; row = 0
  new_aln = np.zeros([msa_length], dtype=np.int32) + msa_depth
  new_mtx = np.zeros([msa_depth, msa_length], dtype=np.int32)
  new_msa = np.zeros([msa_depth, msa_length], dtype=np.int32) + 21
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
  return new_msa, None, new_mtx, new_aln

################################################################################
def create_species_support(
  monomers: List[Dict[str, np.ndarray]]):
  """ Computes relevant information on each species identifier for each chain
    in the target

  Args:
    monomers: a list of dictionaries containing features of a monomer in the multimer target

  Returns:
    common_species: a list of species identifiers present in more than one chain
    all_chain_species_dict: A list of chain_species_dict for each chain, where each
      chain_species_dict is a dictionary mapping <species identifier> to <msa rows>
    num_examples: number of individual chains in the target
  """
  examples = []

  for mono_entry in monomers:
    feats = mono_entry['feature_dict']
    copy_num = mono_entry['copy_number']
    for i in range(copy_num):
      examples.append(feats)

  common_species = set()
  all_chain_species_dict = []

  num_examples = len(examples)

  for f in examples:
      chain_msa = f['msa']

      query_seq = chain_msa[0]
      per_seq_similarity = np.sum(
          query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))

      msa_df = pd.DataFrame({
        'msa_species_identifiers':
            f['msa_species_identifiers'],
        'msa_row':
            np.arange(len(
                f['msa_species_identifiers'])),
        'msa_similarity': per_seq_similarity,
      })
      species_dict = msa_pairing._create_species_dict(msa_df)
      common_species.update(set(species_dict))
      all_chain_species_dict.append(species_dict)

  common_species = sorted(common_species)
  common_species.remove(b'')

  return common_species, all_chain_species_dict, examples

################################################################################
def find_paired_rows(
  common_species: List[str],
  all_chain_species_dict: Dict[int, Dict[bytes, pd.DataFrame]],
  num_examples: int):
  """ Finds MSA species pairings along every chain

  Args:
    common_species: a list of species identifiers present in more than one chain
    all_chain_species_dict: A list of chain_species_dict for each chain, where each
      chain_species_dict is a dictionary mapping <species identifier> to <msa rows>
    num_examples: number of individual chains in the target

  Returns:
    all_paired_msa_rows_dict: dictionary mapping the number of <examples paired> to <the list of
      MSA indices from each chain's MSA dataframe>
  """
  all_paired_msa_rows = [np.zeros(num_examples, int)]
  all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
  all_paired_msa_rows_dict[num_examples] = [np.zeros(num_examples, int)]

  for species in common_species:
      if not species:
        continue
      species_dfs_present = 0
      this_species_msa_dfs = []
      for i, species_dict in enumerate(all_chain_species_dict):
        if (species in species_dict):
          this_species_msa_dfs.append(species_dict[species])
          species_dfs_present += 1
        else:
          this_species_msa_dfs.append(None)
      # Skip species that are present in only one chain.
      if species_dfs_present <= 1:
        continue

      if np.any(
        np.array([len(species_df) for species_df in
            this_species_msa_dfs if
            isinstance(species_df, pd.DataFrame)]) > 600):
        continue
      paired_msa_rows = msa_pairing._match_rows_by_sequence_similarity(this_species_msa_dfs)

      all_paired_msa_rows.extend(paired_msa_rows)
      all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)

  all_paired_msa_rows_dict = {
    num_examples: np.array(paired_msa_rows) for
    num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
  }
  return all_paired_msa_rows_dict

def find_paired_rows_subgraph(
  common_species: List[str],
  all_chain_species_dict: Dict[int, Dict[bytes, pd.DataFrame]],
  num_examples: int,
  subgraphs: List[Tuple[int]]):
  """ For each subgraph, finds all MSA rows for each chain that should be paired.

  Args:
    common_species: a list of species identifiers present in more than one chain
    all_chain_species_dict: A list of chain_species_dict for each chain, where each
      chain_species_dict is a dictionary mapping <species identifier> to <msa rows>
    num_examples: number of individual chains in the target
    subgraphs: a list of subgraphs detailing which group of chains to be paired in the msa features

  Returns:
    all_paired_msa_rows_dict: dictionary mapping the number of <examples paired> to <the list of
      MSA indices from each chain's MSA dataframe>
  """
  all_paired_msa_rows = [np.zeros(num_examples, int)]
  all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
  all_paired_msa_rows_dict[num_examples] = [np.zeros(num_examples, int)]

  already_included_single = {}
  for subgraph in subgraphs:
    for species in common_species:
        if not species:
          continue
        species_dfs_present = set()
        this_species_msa_dfs = [None for _ in range(len(all_chain_species_dict))]
        for i in subgraph:
            species_dict = all_chain_species_dict[i]
            if (species in species_dict):
                this_species_msa_dfs[i] = species_dict[species]
                species_dfs_present.add(i)
        # Skip species that are present in only one chain.

        if len(species_dfs_present) <= 1:
          continue

        if np.any(
          np.array([len(species_df) for species_df in
              this_species_msa_dfs if
              isinstance(species_df, pd.DataFrame)]) > 600):
          continue
        paired_msa_rows = msa_pairing._match_rows_by_sequence_similarity(this_species_msa_dfs)

        all_paired_msa_rows.extend(paired_msa_rows)
        all_paired_msa_rows_dict[len(species_dfs_present)].extend(paired_msa_rows)

  # add in rows without species identifiers
  for i, species_dict in enumerate(all_chain_species_dict):
    this_species_msa_dfs = [None for _ in range(len(all_chain_species_dict))]
    if b'' not in species_dict:
        continue
    this_species_msa_dfs[i] = species_dict[b'']
    paired_msa_rows = msa_pairing._match_rows_by_sequence_similarity(this_species_msa_dfs)
    all_paired_msa_rows_dict[1].extend(paired_msa_rows)

  all_paired_msa_rows_dict = {
    num_examples: np.array(paired_msa_rows) for
    num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
  }
  return all_paired_msa_rows_dict


def msa_pairing_custom(
  monomers: List[Dict[str, np.ndarray]],
  max_template_hits: int,
  subgraphs: List[Tuple[int]] = None,
  pair_all: bool = True):
  """ Main msa pairing code for arbitrary adjacencies between chains. Given a list of subgraphs,
    this code will build the msa features by pairing only the chains within a subgraph.

  Args:
    monomers: a list of dictionaries containing features of a monomer in the multimer target
    max_template_hits: the max number of templates to be included in the features
    subgraphs: a list of subgraphs detailing which group of chains to be paired in the msa features
    pair_all: a override to the subgraphs argument, if True will pair all chains like the AF-Multimer
      paper

  Returns:
    new_msa: the new MSA feature with paired MSAs of each chain
    new_bert_mask: the bert_mask associated with the MSA feature
    new_mtx: the adjusted mtx feature after MSA pairing
    new_aln: the adjusted aln feature after MSA pairing.
  """
  assert (not pair_all and subgraphs is not None) or (pair_all and subgraphs is None), \
    'Inputs should satisfy: pair_all=True and subgraphs=None OR pair_all=False and subgraphs!=None'
  valid_feats = msa_pairing.MSA_FEATURES + (
    'msa_species_identifiers',
  )
  np_chains_list = []
  for i in range(len(monomers)):
    orig_feats = monomers[i]['feature_dict']
    new_feats = {}
    for k, v in orig_feats.items():
        if k in valid_feats:
            new_feats[f'{k}_all_seq'] = v
    monomers[i]['feature_dict'].update(new_feats)

    np_chains_list.append(monomers[i]['feature_dict'])

  common_species, all_chain_species_dict, examples = create_species_support(monomers)
  if pair_all:
    all_paired_msa_rows_dict = find_paired_rows(
      common_species, all_chain_species_dict, len(examples))
  else:
    all_paired_msa_rows_dict = find_paired_rows_subgraph(
      common_species, all_chain_species_dict, len(examples), subgraphs)
  paired_rows = msa_pairing.reorder_paired_rows(all_paired_msa_rows_dict)

  chain_keys = list(np_chains_list[0].keys())

  updated_chains = []
  for i, chain in enumerate(np_chains_list):
    new_chain = {k: v for k, v in chain.items() if '_all_seq' not in k}
    for feature_name in chain_keys:
      if feature_name.endswith('_all_seq'):
        feats_padded = msa_pairing.pad_features(chain[feature_name], feature_name)
        new_chain[feature_name] = feats_padded[paired_rows[:, i]]
    new_chain['num_alignments_all_seq'] = np.asarray(
        len(paired_rows[:, i]))
    updated_chains.append(new_chain)

  chains = msa_pairing.deduplicate_unpaired_sequences(updated_chains)
  example = msa_pairing.merge_chain_features(chains, True, max_template_hits)

  new_msa = example['msa']
  new_bert_mask = example['bert_mask']
  new_mtx = example['deletion_matrix_int']
  new_aln = example['num_alignments']

  return new_msa, new_bert_mask, new_mtx, new_aln

################################################################################
def template_cropping_and_joining_mono(curr_input: Dict):
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

  new_tem = initialize_template_feats(full_num_tem, full_num_res, False)
  col = 0; row = 0
  for mono_entry in monomers:
    features = extract_template_domain_mono(mono_entry)

    copy_num = mono_entry['copy_number']
    #num_res = features['template_aatype'].shape[1]
    num_res = features['msa'].shape[1]
    num_tem = len(features['template_domain_names'])

    for i in range(copy_num):
      col_ = col + num_res
      row_ = row + num_tem

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
  
################################################################################
def template_cropping_and_joining_mult(curr_input: Dict):
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

  new_tem = initialize_template_feats(full_num_tem, full_num_res, is_multimer=True)
  for mono_entry in monomers:
    features = extract_template_domain_mult(mono_entry)
    copy_num = mono_entry['copy_number']

    #num_res = features['template_aatype'].shape[1]
    num_res = features['msa'].shape[1]
    num_tem = len(features['template_domain_names'])

    for i in range(copy_num):
      col_ = col + num_res
      row_ = row + num_tem

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
