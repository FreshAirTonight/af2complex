# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is a modification of DeepMind's run_alphafold.py
#
# Run AlphaFold DL modeld inference to predict protein (complex) structure
# Input: pre-generated features from AF2 data pipeline such as by run_af2c_fea.py
# Output: protein structure models
#
# Note: AF2Complex is a modified, enhanced version of AlphaFold 2.
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology, 2021-2023
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

from absl import app
from absl import flags
from absl import logging

from alphafold.common import protein
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data import pipeline

from alphafold.data.complex import *
from datetime import date

import numpy as np
# Internal import (7716).


flags.DEFINE_string('target_lst_path', None, 'Path to a file containing a list of targets '
                  'in any monomer, homo- or hetero-oligomers '
                  'configurations. For example, TarA is a monomer, TarA:2 is a dimer '
                  'of two TarAs. TarA:2/TarB is a trimer of two TarA and one TarB, etc.'
                  )
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('feature_dir', None, 'Path to a directory that will '
                    'contains pre-genearted feature in pickle format.')
flags.DEFINE_list('model_names', None, 'Names of deep learning models to use. This is required even with model_preset option.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_enum('preset', None,
                  ['deepmind', 'casp14', 'economy', 'super', 'expert', 'super2', 'genome', 'genome2'],
                  'Choose preset model configuration: <deepmind> standard settings according to DeepMind, '
                  ' i.e., 3 recycles and 1 ensemble; '
                  '<economy> no ensemble, up to 256 MSA clusters, recycling up to 3 rounds; '
                  '<super, super2> 1 or 2 ensembles, up to 512 MSA clusters, recycling up to 20 rounds; '
                  '<genome, genome2> 1 or 2 ensembles, up to 512 MSA clusters, max number '
                  'of recycles and ensembles adjusted according to input sequence length; '
                  '<expert> similar to super but maintain the same recycle number regardless target size; '
                  'or <casp14> 8 model ensemblings used by DeepMind in CASP14.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('max_recycles', None, 'The maximum number of recycles.', lower_bound=1)
flags.DEFINE_float('recycle_tol', None, 'The tolerance for recycling, caculated as the RMSD change '
                    'in the distogram of backbone Ca atoms. Recycling stops '
                    'if the change is smaller than this value', lower_bound=0.0, upper_bound=2.0)
flags.DEFINE_integer('num_ensemble', None, 'The number of ensembles of each model, 1 means no ensembling.', lower_bound=1)
flags.DEFINE_integer('max_msa_clusters', None, 'The maximum number of MSA clusters.', lower_bound=1)
flags.DEFINE_integer('max_extra_msa', None, 'The maximum number of extra MSA clusters.', lower_bound=1)
flags.DEFINE_boolean('write_complex_features', False, 'Save the feature dict for '
                    'complex prediction as a pickle file under the output direcotry')
flags.DEFINE_boolean('no_template', False, 'Do not use structural template. Note that '
                  'this does not have an impact on models that do not use template regardless.')
flags.DEFINE_boolean('output_pickle', True, 'Write the prediction results into a pickle file. '
                  'Note that the pickle files are quite large. Disable it to save disk space.')
flags.DEFINE_integer('save_recycled', 0, '0 - no recycle info saving, 1 - print '
                   'metrics of intermediate recycles, 2 - additionally saving pdb structures '
                   'of all recycles, 3 - additionally save all results in pickle '
                    'dictionaries of each recycling iteration.', lower_bound=0, upper_bound=3)
flags.DEFINE_string('checkpoint_tag', None, 'Enable checkpoint and use the tag to name '
                    'files to restart the recycle modeling later.')
flags.DEFINE_integer('max_mono_msa_depth', 10000, 'The maximum MSA depth for each monomer', lower_bound=1)
flags.DEFINE_integer('mono_msa_crop_size', 5000, 'For monomer MSA cropping in the multimer_np mode', lower_bound=1)
flags.DEFINE_integer('max_template_hits', 4, 'The maximum PDB template for each monomer', lower_bound=0)
flags.DEFINE_enum('model_preset', 'monomer_ptm',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer', 'multimer_np'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head (monomer_ptm), multimer on multimer features with paired MSA generated by AF-Multimer data pipeline '
                  '(multimer), and multimer model on monomer features and unpaired MSAs (mulitmer_np).')
flags.DEFINE_integer('num_predictions_per_model', 1, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. ', lower_bound=1)
flags.DEFINE_enum('msa_pairing', None,
                  ['all', 'cyclic', 'linear','custom'],
                  'Choose MSA pairing mode if using input features of monomers - By default no action, '
                  'all - pairing as many as possible, the most dense complex MSAs, '
                  'cyclic - sequentially pairing the nearest neighbor defined in the stoichoimetry, '
                  'custom - use a defined list of pairs.')
flags.DEFINE_boolean('do_cluster_analysis', False, 'Whether to print out clusters of protein chains in the prediction')
flags.DEFINE_integer('cluster_edge_thres', 10, 'The number of contacts between chains that constitute an edge in the '
                  'cluster analysis', lower_bound=0)
flags.DEFINE_float('pdb_iscore_cf', -1.0, 'If interface icore is present, only write the pdb of the structural model '
                    'if the iScore is larger than this cutoff value. Useful for large-scale screening. ')
flags.DEFINE_boolean('allow_dropout', False, 'Allow dropout during model inference. This is an experimental feature. '
                     'Default is disabled.')

FLAGS = flags.FLAGS

Flag = Type[FLAGS]



##################################################################################################
def load_checkpoint_file(ckpt_file_path: str):
    with open(ckpt_file_path, "rb") as f:
        prev, prev_ckpt_iter = pickle.load(f)
        f.close()
        logging.info(f"Previous checkpoint was saved at recycle {prev_ckpt_iter}, continuing...")
    return prev, prev_ckpt_iter

def get_asymid2chain_name(target):
  idx2chain_name = []
  # for monomer in target['split']:
  #   idx2chain_name.extend([monomer['mono_name'] for i in range(monomer['copy_number'])])
  counter = {}
  for monomer in target['split']:
    n = monomer['mono_name']
    for i in range(monomer['copy_number']):
      if n in counter:
        idx2chain_name.append(f'{n}_{counter[n]}')
        counter[n] += 1
      else:
        idx2chain_name.append(f'{n}_{0}')
        counter[n] = 1

  if target['asym_id_list'] is not None:
    new_idx2chain_name = []
    for id_list in target['asym_id_list']:
      new_idx2chain_name.append('/'.join([idx2chain_name[i] for i in id_list]))
    idx2chain_name = new_idx2chain_name
  return idx2chain_name
##################################################################################################

##################################################################################################
def predict_structure(
    target: Dict[str, str],
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    max_msa_clusters: int,
    max_extra_msa: int,
    max_recycles: int,
    num_ensemble: int,
    flags: Flag):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}

  target_name = target['name']
  asym_id_list = target['asym_id_list']
  is_multimer = "multimer" in flags.model_preset

  # Retrieve pre-generated features of monomers (single protien sequences)
  t_0 = time.time()
  monomers = target['split']
  # create feature dictionary for a monomeric or multimeric target
  feature_dict, mono_chains, Ls = make_complex_features(target, flags)
  print(f"Info: {len(mono_chains)} chain(s) to model {mono_chains}")

  N = len(feature_dict["msa"])
  L = len(feature_dict["residue_index"])
  T = 0
  if not flags.no_template: T = feature_dict["template_all_atom_positions"].shape[0]
  print(f"Info: modeling {target_name} with msa_depth = {N}, seq_len = {L}, num_templ = {T}")
  timings['features'] = round(time.time() - t_0, 2)

  output_dir = os.path.join(flags.output_dir, target_name)
  if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:  # this could happen when multiple runs are working on the same target simultaneously
        print(f"Warning: tried to create an existing {output_dir}, ignored")

  if flags.write_complex_features:
      feature_output_path = os.path.join(output_dir, 'features_comp.pkl')
      with open(feature_output_path, 'wb') as f:
          pickle.dump(feature_dict, f, protocol=4)

  today = date.today().strftime('%y%m%d')
  out_suffix = '_' + today + '_' + str(random_seed)[-6:]

  plddts = {}  # predicted LDDT score
  iterations = {}  # recycle information
  ptms = {}; pitms = {} # predicted TM-score
  ires = {}; icnt = {} # interfacial residues and contacts
  tols = {} # change in backbone pairwise distance to check with the recyle tolerance criterion
  ints = {} # interface-score
  iptms = {} # this is DeepMind's predicted interface TM-score
  clus = {} # cluster analysis results
  # Run models for structure prediction
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    model_random_seed = model_index + random_seed
    model_out_name = model_name + out_suffix
    logging.info('Running model %s', model_out_name)

    prev_ckpt = None; prev_ckpt_iter = 0
    model_runner.checkpoint_file = None
    if flags.checkpoint_tag:
        checkpoint_dir  = os.path.join(output_dir, 'checkpoint')
        model_runner.checkpoint_file = os.path.join(checkpoint_dir, model_name + '_' + flags.checkpoint_tag + ".pkl")
        if os.path.exists(model_runner.checkpoint_file):
            logging.info(f"Found checkpoint file {model_runner.checkpoint_file}, loading...")
            prev_ckpt, prev_ckpt_iter = load_checkpoint_file(model_runner.checkpoint_file)
            # model_runner.predict will run max_recycles + 1 rounds of model inference,
            # when restart from previous checkpoint, first round inference should count as one recycle.
            if not is_multimer:
                model_runner.config.data.common.num_recycle = max_recycles - 1
            model_runner.config.model.num_recycle = max_recycles - 1

    t_0 = time.time()
    # set size of msa (to reduce memory requirements)
    if max_msa_clusters is not None and max_extra_msa is not None and not is_multimer:
        msa_clusters = max(min(N, max_msa_clusters),5)
        model_runner.config.data.eval.max_msa_clusters = msa_clusters
        model_runner.config.data.common.max_extra_msa = max(min(N-msa_clusters,max_extra_msa),1)
    if flags.preset in ['genome', 'genome2', 'super']:
        max_iter = max_recycles - max(0, (L - 500) // 50)
        max_iter = max( 6, max_iter )
        if L > 1180:    ### memory limit of a single 16GB GPU, applied to cases with multiple ensembles
            num_en = 1
        else:
            num_en = num_ensemble
        print(f"Info: {target_name} reset max_recycles = {max_iter}, num_ensemble = {num_en}")
        if not is_multimer:
            model_runner.config.data.common.num_recycle = max_iter
            model_runner.config.data.eval.num_ensemble = num_en
        model_runner.config.model.num_recycle = max_iter

    asym_id = feature_dict['asym_id']
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_out_name}'] = round(time.time() - t_0, 2)

    t_0 = time.time()
    prediction_result, (tot_recycle, tol_value, recycled) = model_runner.predict(processed_feature_dict,
            random_seed=model_random_seed, prev=prev_ckpt, prev_ckpt_iter=prev_ckpt_iter,
            asym_id_list=asym_id_list, asym_id=asym_id, edge_contacts_thres=FLAGS.cluster_edge_thres)
    tot_recycle += prev_ckpt_iter
    if prev_ckpt_iter: print(f"Info: total recycle number is {tot_recycle}")
    prediction_result['num_recycle'] = tot_recycle
    prediction_result['mono_chains'] = mono_chains

    tols[model_out_name] = round(tol_value.tolist(), 3)
    iterations[model_out_name] = tot_recycle.tolist()  ### convert jax numpy to regular list for json saving

    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_out_name}'] = round(t_diff,1)
    logging.info(
      'Total JAX model %s predict time (includes compilation time): %.1f seconds', model_out_name, t_diff)

    def _save_results(result, log_model_name, out_dir, recycle_index):
      # Get mean pLDDT confidence metric.
      plddt = np.mean(result['plddt'])
      plddts[log_model_name] = round(plddt, 2)
      ptm = 0; iptm = 0
      if 'ptm' in result:
          ptm = result['ptm'].tolist()
          ptms[log_model_name] = round(ptm, 4)
      pitm = 0; inter_residues = 0; inter_contacts = 0; inter_sc = 0
      if 'pitm' in result:
          pitm = result['pitm']['score'].tolist()
          inter_residues = result['pitm']['num_residues'].tolist()
          inter_contacts = result['pitm']['num_contacts'].tolist()
          pitms[log_model_name] = round(pitm, 4)
          ires[log_model_name] = inter_residues
          icnt[log_model_name] = int(inter_contacts)
      if 'interface' in result:
          inter_sc = result['interface']['score'].tolist()
          ints[log_model_name] = round(inter_sc, 4)
      if 'iptm+ptm' in result:
          iptm = result['iptm+ptm'].tolist()
          iptms[log_model_name] = round(iptm, 4)

      if recycle_index < tot_recycle:
          tol = result['tol_val'].tolist()
          tols[log_model_name] = round(tol, 2)
          print(f"Info: {target_name} {log_model_name}, ",
            f"tol = {tol:5.2f}, pLDDT = {plddt:.2f}, pTM-score = {ptm:.4f}", end='')
          if len(monomers) > 1 or monomers[0]['copy_number'] > 1: # hetero- or homo-oligomer target
            print(f", piTM-score = {pitm:.4f}, interface-score = {inter_sc:.4f}",
              f", iRes = {inter_residues:<4d} iCnt = {inter_contacts:<4.0f}")
          else:
            print('')
      else:
          print(f"Info: {target_name} {log_model_name} performed {tot_recycle} recycles,",
            f"final tol = {tol_value:.2f}, pLDDT = {plddt:.2f}", end='')
          if 'iptm+ptm' in result:
              print(f", iptm+ptm = {iptm:.4f}", end='')
          else:
              print(f", pTM-score = {ptm:.4f}", end='')
          if len(monomers) > 1 or monomers[0]['copy_number'] > 1:            
            print(f", piTM-score = {pitm:.4f}, interface-score = {inter_sc:.4f}",
                f", iRes = {inter_residues:<4d} iCnt = {inter_contacts:<4.0f}")
            if 'cluster_analysis' in result:
              clus_res = result['cluster_analysis']
              idx2chain_name = get_asymid2chain_name(target)
              cluster_identities = []
              for cluster in clus_res['clusters']:
                cluster_identities.append([idx2chain_name[c] for c in cluster])
              clus[log_model_name] = {
                'num_clusters': clus_res['num_clusters'],
                'cluster_sizes': clus_res['cluster_size'],
                'clusters': cluster_identities,
              }
              print(f"Info: num_clusters = {clus_res['num_clusters']}, cluster_sizes = {clus_res['cluster_size']}, ",
                  f"clusters = {cluster_identities}\n")
          else:
            print('')

      # Save the model outputs, not saving pkl for intermeidate recycles to save storage space
      # skip saving pkl if iScore less than the specified cutoff
      if ((recycle_index == tot_recycle and flags.output_pickle) or FLAGS.save_recycled == 3) and inter_sc > FLAGS.pdb_iscore_cf:
          result_output_path = os.path.join(out_dir, f'{log_model_name}.pkl')
          with open(result_output_path, 'wb') as f:
              pickle.dump(result, f, protocol=4)

      if (recycle_index == tot_recycle or FLAGS.save_recycled >= 2) and inter_sc > FLAGS.pdb_iscore_cf:
          # Set the b-factors to the per-residue plddt
          final_atom_mask = result['structure_module']['final_atom_mask']
          b_factors = result['plddt'][:, None] * final_atom_mask

          unrelaxed_protein = protein.from_prediction(feature_dict,
                                      result,
                                      b_factors=b_factors,
                                      remove_leading_feature_dimension=not is_multimer,
                                      is_multimer=is_multimer,
                                      )

          unrelaxed_pdb_path = os.path.join(out_dir, f'{log_model_name}.pdb')
          with open(unrelaxed_pdb_path, 'w') as f:
            if flags.model_preset == 'multimer_np':
                f.write(protein.to_pdb(unrelaxed_protein, False))
            else:
                f.write(protein.to_pdb(unrelaxed_protein, is_multimer))

    # output info of intermeidate recycles and save the coordinates
    if FLAGS.save_recycled:
      recycle_out_dir = os.path.join(output_dir, "recycled")
      if FLAGS.save_recycled > 1 and not os.path.exists(recycle_out_dir):
          try:
            os.mkdir(recycle_out_dir)
          except FileExistsError:  # this could happen when multiple runs are working on the same target simultaneously
            print(f"Warning: tried to create an existing {recycle_out_dir}, ignored")

      for rec_idx, rec_dict in enumerate(recycled):
        if prev_ckpt_iter: rec_idx += prev_ckpt_iter
        if rec_idx < tot_recycle:
            _save_results(rec_dict, f"{model_out_name}_recycled_{rec_idx:02d}",
                          recycle_out_dir, rec_idx)

    # the final results from this model run
    _save_results(prediction_result, model_out_name, output_dir, tot_recycle)
  # End of model runs

  # Rank by pTMscore if exists, otherwise pLDDTs
  ranked_order = []
  if 'ptm' in prediction_result:
      ranking_metric = 'pTM'
      for idx, (mod_name, _) in enumerate(
        sorted(ptms.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(mod_name)
  else:
      ranking_metric = 'pLDDT'
      for idx, (mod_name, _) in enumerate(
        sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(mod_name)

  stats = {'plddts': plddts, 'ptms': ptms, 'order': ranked_order,
        'ranking_metric': ranking_metric, 'iterations': iterations,
        'tol_values':tols, 'chains': mono_chains, 'chain_lengths': Ls,
        'timings':timings, 'model_preset': FLAGS.model_preset,
        'msa_pairing':FLAGS.msa_pairing}

  if len(pitms):  ## extra metrics for evaluating interactions
      stats = { **stats, 'pitms': pitms, 'interfacial residue number': ires,
            'interficial contact number': icnt, 'interface score': ints,
            'iptm+ptm': iptms, 'clusters': clus }

  if len(model_runners) > 1:  #more than 1 model
      ranking_output_path = os.path.join(output_dir, 'ranking_all'+out_suffix+'.json')
  else: #only one model, use different model names to avoid overwriting same file
      ranking_output_path = os.path.join(output_dir, 'ranking_'+model_out_name+'.json')

  with open(ranking_output_path, 'w') as f:
    f.write(json.dumps(stats, sort_keys=True, indent=4))

  logging.info('Final timings for %s: %s', target_name, timings)

##################################################################################################

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # read a list of target files
  target_lst = read_af2c_target_file( FLAGS.target_lst_path )

  max_recycles = 3; recycle_tol = 0; num_ensemble = 1
  max_extra_msa = None; max_msa_clusters = None
  print("Info: using preset", FLAGS.preset)
  if FLAGS.preset == 'casp14':
    num_ensemble = 8
  elif FLAGS.preset == 'economy':
    recycle_tol  = 0.1
    max_extra_msa = 512
    max_msa_clusters = 256
  elif FLAGS.preset in ['super', 'expert', 'super2']:
    max_recycles = 20
    recycle_tol  = 0.1
    max_extra_msa = 1024
    max_msa_clusters = 512
    if FLAGS.preset == 'super2': num_ensemble = 2
  elif FLAGS.preset in ['genome', 'genome2']:
    max_recycles = 20
    recycle_tol  = 0.5
    max_extra_msa = 1024
    max_msa_clusters = 512
    if FLAGS.preset == 'genome2': num_ensemble = 2

  # allow customized parameters over preset
  if FLAGS.num_ensemble is not None:
    num_ensemble = FLAGS.num_ensemble
  print(f"Info: set num_ensemble = {num_ensemble}")
  if FLAGS.max_recycles is not None:
    max_recycles = FLAGS.max_recycles
  print(f"Info: set max_recyles = {max_recycles}")
  if FLAGS.recycle_tol is not None:
    recycle_tol = FLAGS.recycle_tol
  print(f"Info: set recycle_tol = {recycle_tol}")
  if FLAGS.max_msa_clusters is not None and FLAGS.max_extra_msa is not None:
    max_msa_clusters = FLAGS.max_msa_clusters
    max_extra_msa = FLAGS.max_extra_msa
    print(f"Info: max_msa_clusters = {max_msa_clusters}, max_extra_msa = {max_extra_msa}")

  if FLAGS.msa_pairing:
    print(f"Info: mas_pairing mode is {FLAGS.msa_pairing}")

  def make_model_runners(model_names, model_preset):
    model_runners = {}
    for model_name in model_names:
      # sanity check model names and model preset
      if 'monomer' in model_preset and 'multimer' in model_name:
          raise ValueError(f"{model_name} is not compatible with {FLAGS.model_preset}" )
      elif 'multimer' in model_preset and not 'multimer' in model_name:
          raise ValueError(f"{model_name} is not compatible with {FLAGS.model_preset}" )

      model_config = config.model_config(model_name)
      if 'multimer' not in model_preset:
          model_config.num_ensemble_eval = num_ensemble
          model_config.data.common.num_recycle = max_recycles
      model_config.model.num_recycle = max_recycles
      model_config.model.recycle_tol = recycle_tol
      model_config.model.save_recycled = FLAGS.save_recycled

      # allow drop out for model inference, this is an advanced feature only for expert users
      if FLAGS.allow_dropout:
          print("Info: allow dropout for model inference")
          model_config.model.global_config.eval_dropout = True

      model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=FLAGS.data_dir)
      model_runner = model.RunModel(model_config, model_params)

      for i in range(FLAGS.num_predictions_per_model):
        model_runners[f'{model_name}_p{i+1}'] = model_runner
    return model_runners

  model_runners = make_model_runners(FLAGS.model_names, FLAGS.model_preset)
  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  time.sleep(random.randint(0,10)) # mitigating creating the same output directory from multiple runs

  # Predict structure for each target.
  for target in target_lst:
    target_name  = target['name']
    target_split = target['split']
    model_names = target['model']
    model_preset = target['model_preset']

    # check if model, model_preset, or msa_pairing are overriden
    if (FLAGS.model_names != model_names or FLAGS.model_preset != model_preset) and \
        (model_names is not None and model_preset is not None):
      model_runners = make_model_runners(model_names, model_preset)
      FLAGS.model_names = model_names
      FLAGS.model_preset = model_preset

    if FLAGS.msa_pairing != target['msa_pairing'] and target['model'] is not None:
      FLAGS.msa_pairing = target['msa_pairing']

    if target['adj_list_path'] is not None and FLAGS.msa_pairing != 'custom':
      print(f'WARNING: adjacency list provided for {target_name}, '
          f'but msa_pairing is set to {FLAGS.msa_pairing}. AF2Complex will not use adjacency list.')

    print(f"Info: working on target {target_name}")
    if FLAGS.random_seed is not None:
        random_seed = FLAGS.random_seed
    else:
        random_seed = random.randrange(sys.maxsize)
    logging.info('Using random seed %d for the data pipeline', random_seed)
    predict_structure(
        target=target,
        model_runners=model_runners,
        random_seed=random_seed,
        max_msa_clusters=max_msa_clusters,
        max_extra_msa=max_extra_msa,
        max_recycles=max_recycles,
        num_ensemble=num_ensemble,
        flags=FLAGS,
        )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'target_lst_path',
      'output_dir',
      'feature_dir',
      'model_names',
      'data_dir',
      'preset'
  ])

  app.run(main)
