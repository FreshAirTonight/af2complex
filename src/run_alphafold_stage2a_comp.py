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
# Run AlphaFold DL modelds using pre-generated features w/o model relaxation
#
# Input: pre-generated features from AlphaFold DataPipeline on single sequences
# Output: un-relaxed protein models
#
# Note: AF2Complex is a modified, enhanced version of AlphaFold 2.
# Additional unofficial features added:
#
# Predicting models of a protein complex including both homooligomer and heterooligomer
# No MSA pairing required
# New metrics designed for evaluating protein-protein interface
# Saving structure models of all recycles
# Split feature generation (stage 1), DL inference (stage 2a), and model relaxation (stage 2b)
#
# Some other features such as option for dynamically controled number of recycles and
# residue index breaks were taken from ColabFold (https://github.com/sokrypton/ColabFold)
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology

"""Enhanced AlphaFold Stage2a: protein complex structure prediction with deep learning"""
import json
import os
import pickle
import random
import sys
import time
import re
from typing import Dict

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
flags.DEFINE_list('model_names', None, 'Names of deep learning models to use.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_enum('preset', None,
                  ['reduced_dbs', 'casp14', 'economy', 'super', 'super2', 'genome', 'genome2'],
                  'Choose preset model configuration: <reduced_dbs> no ensembling, '
                  '<economy> no ensemble, up to 256 MSA clusters, recycling up to 3 rounds; '
                  '<super, super2> 1 or 2 ensembles, up to 512 MSA clusters, recycling up to 20 rounds; '
                  '<genome, genome2> 1 or 2 ensembles, up to 512 MSA clusters, max number '
                  'of recycles and ensembles adjusted according to input sequence length; '
                  'or <casp14> 8 model ensemblings of the factory settings.')
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
flags.DEFINE_enum('template_mode', 'oligomer', ['none', 'monomer', 'oligomer'],
                  'none - No template is allowed, '
                  'monomer - Use template only for monomer but not oligomer modeling, '
                  'oligomer - Use monomer template for all modeling if exists.')
flags.DEFINE_integer('save_recycled', 0, '0 - no recycle info saving, 1 - print '
                   'metrics of intermediate recycles, 2 - additionally saving pdb structures '
                   'of all recycles, 3 - additionally save all results in pickle '
                    'dictionaries of each recycling iteration.', lower_bound=0, upper_bound=3)

FLAGS = flags.FLAGS

#MAX_TEMPLATE_HITS = 20
MAX_MSA_DEPTH_MONO = 50000   ### maximum number of input sequences in the msa of a monomer

##################################################################################################
# read either a single target string or a input list of targets in a file
# each line has a format like: <target> <length> (output_name), the output_name is optional.
# In <target>, use monomer:num to indicate num of copies in a homooligomer
# and name1/name2 to indicate heterooligomer. For example, TarA:2/TarB is 2 copies of TarA and 1 TarB
def _read_target_file( data_lst_file ):
    target_lst = []
    if not os.path.exists( data_lst_file ):  ### input is a single target in strings
        fields = data_lst_file.split(',')
        fullname = name = fields[0]
        if len(fields) == 2:
            name = fields[1]
        target_lst.append( {'full':fullname, 'name':name} )
    else: ### input are a list of targets in a file
      with open( data_lst_file ) as file:
        for line in file:
            if line.startswith("#"):
                continue
            line = line.strip() # strip "\n"
            fields = line.split()
            fullname = name = fields[0]
            if len(fields) > 2 and not fields[2].startswith("#"):
                name = fields[2]
            target_lst.append( {'full':fullname, 'name':name} )

    # process the components of a complex if detected
    for target in target_lst:
        complex = target['full']
        monomers = []
        subfields = complex.split('/')
        for item in subfields:
            cols = item.split(':')
            if len(cols) == 1:
                monomers.append( {cols[0]:1} )  ### monomer
            elif len(cols) > 1:
                monomers.append( {cols[0]:cols[1]} )
        if len(monomers) >= 1:
            target['split'] = monomers

    return target_lst
##################################################################################################


##################################################################################################
def predict_structure(
    target: Dict[str, str],
    output_dir_base: str,
    feature_dir_base: str,
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    max_msa_clusters: int,
    max_extra_msa: int,
    max_recycles: int,
    num_ensemble: int,
    preset: str,
    write_complex_features: bool,
    template_mode: str):
  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}

  target_name  = target['name']
  target_name = re.sub(":", "_x", target_name)
  target_name = re.sub("/", "+", target_name)

  homo_copy = []
  seq_names = []
  for homo in target['split']:
    for seq_name, seq_copy in homo.items():
      homo_copy.append(int(seq_copy))
      seq_names.append(seq_name)

  time.sleep(random.randint(0,30)) # mitigating creating the same output directory from multiple runs

  # Retrieve pre-generated features of monomers (single protien sequences)
  t_0 = time.time()
  feature_dicts = []
  for seq_name in seq_names:
    feature_dir = os.path.join(feature_dir_base, seq_name)
    if not os.path.exists(feature_dir):
      raise SystemExit("Error: ", feature_dir, "does not exists")

    # load pre-generated features as a pickled dictionary.
    features_input_path = os.path.join(feature_dir, 'features.pkl')
    with open(features_input_path, "rb") as f:
      mono_feature_dict = pickle.load(f)
      N = len(mono_feature_dict["msa"])
      L = len(mono_feature_dict["residue_index"])
      T = len(mono_feature_dict["template_domain_names"])
      print(f"Info: {target_name} found monomer {seq_name} msa_depth = {N}, seq_len = {L}, num_templ = {T}")
      if N > MAX_MSA_DEPTH_MONO:
          print(f"Info: {seq_name} MSA size is too large, reducing to {MAX_MSA_DEPTH_MONO}")
          mono_feature_dict["msa"] = mono_feature_dict["msa"][:MAX_MSA_DEPTH_MONO,:]
          mono_feature_dict["deletion_matrix_int"] = mono_feature_dict["deletion_matrix_int"][:MAX_MSA_DEPTH_MONO,:]
          mono_feature_dict['num_alignments'][:] = MAX_MSA_DEPTH_MONO
      feature_dicts.append( mono_feature_dict )

  # Make features for complex structure prediction using monomer structures if necessary
  if len(seq_names) == 1 and homo_copy[0] == 1:   # monomer structure prediction
    feature_dict = feature_dicts[0]
    seq_len = len(feature_dict["residue_index"])
    Ls = [seq_len]
    if template_mode == 'none':
        new_tem = initialize_template_feats(0, seq_len)
        feature_dict.update(new_tem)
  else:  # complex structure prediction
    feature_dict, Ls = make_complex_features(feature_dicts, target_name, homo_copy, template_mode)

  mono_chains = []
  mono_chains = get_mono_chain(seq_names, homo_copy, Ls)
  print(f"Info: individual chain(s) to model {mono_chains}")

  N = len(feature_dict["msa"])
  L = len(feature_dict["residue_index"])
  T = len(feature_dict["template_domain_names"])
  print(f"Info: modeling {target_name} with msa_depth = {N}, seq_len = {L}, num_templ = {T}")
  timings['features'] = round(time.time() - t_0, 2)


  output_dir = os.path.join(output_dir_base, target_name)
  if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print(f"Warning: tried to create an existing {output_dir}, ignored")

  if write_complex_features:
      feature_output_path = os.path.join(output_dir, 'features_comp.pkl')
      with open(feature_output_path, 'wb') as f:
          pickle.dump(feature_dict, f, protocol=4)

  today = date.today().strftime('%Y%m%d')
  out_suffix = '_' + today + '_' + str(random_seed)[-6:]

  plddts = {}  # predicted LDDT score
  iterations = {}  # recycle information
  ptms = {}; pitms = {} # predicted TM-score
  ires = {}; icnt = {} # interfacial residues and contacts
  tols = {} # change in backbone pairwise distance to check with the recyle tolerance criterion
  ints = {} # interface-score
  # Run models for structure prediction
  for model_name, model_runner in model_runners.items():
    model_out_name = model_name + out_suffix
    logging.info('Running model %s', model_out_name)
    t_0 = time.time()

    # set size of msa (to reduce memory requirements)
    if max_msa_clusters is not None and max_extra_msa is not None:
        msa_clusters = max(min(N, max_msa_clusters),5)
        model_runner.config.data.eval.max_msa_clusters = msa_clusters
        model_runner.config.data.common.max_extra_msa = max(min(N-msa_clusters,max_extra_msa),1)
    if preset in ['genome', 'genome2', 'super']:
        max_iter = max_recycles - max(0, (L - 500) // 50)
        max_iter = max( 6, max_iter )
        if L > 1180:    ### memory limit of a single 16GB GPU, applied to cases with mutliple ensembles
            num_en = 1
        else:
            num_en = num_ensemble
        print(f"Info: {target_name} reset max_recycles = {max_iter}, num_ensemble = {num_en}")
        model_runner.config.data.common.num_recycle = max_iter
        model_runner.config.model.num_recycle = max_iter
        model_runner.config.data.eval.num_ensemble = num_en

    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    timings[f'process_features_{model_out_name}'] = round(time.time() - t_0, 2)

    t_0 = time.time()
    prediction_result, (tot_recycle, tol_value, recycled) = model_runner.predict(processed_feature_dict)
    prediction_result['num_recycle'] = tot_recycle
    prediction_result['mono_chains'] = mono_chains

    tols[model_out_name] = round(tol_value.tolist(), 3)
    iterations[model_out_name] = tot_recycle.tolist()  ### convert from jax numpy to regular list for json saving

    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_out_name}'] = round(t_diff,1)
    logging.info(
      'Total JAX model %s predict time (includes compilation time): %.1f seconds', model_out_name, t_diff)

    def _save_results(result, log_model_name, out_dir, recycle_index):
      # Get mean pLDDT confidence metric.
      plddt = np.mean(result['plddt'])
      plddts[log_model_name] = round(plddt, 2)
      ptm = 0
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

      if recycle_index < tot_recycle:
          tol = result['tol_val'].tolist()
          tols[log_model_name] = round(tol, 2)
          print(f"Info: {target_name} {log_model_name}, ",
            f"tol = {tol:5.2f}, pLDDT = {plddt:.2f}, pTM-score = {ptm:.4f}", end='')
          if len(seq_names) > 1 or sum(homo_copy) > 1: # complex target
            print(f", piTM-score = {pitm:.4f}, interface-score = {inter_sc:.4f}", end='')
            print(f", iRes = {inter_residues:<4d} iCnt = {inter_contacts:<4.0f}")
          else:
            print('')
      else:
          print(f"Info: {target_name} {log_model_name} performed {tot_recycle} recycles,",
            f"final tol = {tol_value:.2f}, pLDDT = {plddt:.2f}, pTM-score = {ptm:.4f}", end='')
          if len(seq_names) > 1 or sum(homo_copy) > 1:
            print(f", piTM-score = {pitm:.4f}, interface-score = {inter_sc:.4f}", end='')
            print(f", iRes = {inter_residues:<4d} iCnt = {inter_contacts:<4.0f}")
          else:
            print('')

      # Save the model outputs, not saving pkl for intermeidate recycles to save storage space
      if recycle_index == tot_recycle or FLAGS.save_recycled == 3:
          result_output_path = os.path.join(out_dir, f'{log_model_name}.pkl')
          with open(result_output_path, 'wb') as f:
              pickle.dump(result, f, protocol=4)

      if recycle_index == tot_recycle or FLAGS.save_recycled >= 2:
          # Set the b-factors to the per-residue plddt
          final_atom_mask = result['structure_module']['final_atom_mask']
          b_factors = result['plddt'][:, None] * final_atom_mask

          unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                result, b_factors=b_factors)

          unrelaxed_pdb_path = os.path.join(out_dir, f'{log_model_name}.pdb')
          with open(unrelaxed_pdb_path, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))

    # output info of intermeidate recycles and save the coordinates
    if FLAGS.save_recycled:
      recycle_out_dir = os.path.join(output_dir, "recycled")
      if FLAGS.save_recycled > 1 and not os.path.exists(recycle_out_dir):
        os.mkdir(recycle_out_dir)
      for i, rec_dict in enumerate(recycled):
        if i < tot_recycle:
            _save_results(rec_dict, f"{model_out_name}_recycled_{i:02d}",
                          recycle_out_dir, i)

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
        'timings':timings}
  if len(pitms):
      stats = { **stats, 'pitms': pitms, 'interfacial residue number': ires,
            'interficial contact number': icnt, 'interface score': ints }

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
  target_lst = _read_target_file( FLAGS.target_lst_path )

  max_recycles = 3; recycle_tol = 0
  max_extra_msa = None; max_msa_clusters = None
  print("Info: using preset", FLAGS.preset)
  if FLAGS.preset == 'reduced_dbs':
    num_ensemble = 1
  elif FLAGS.preset == 'casp14':
    num_ensemble = 8
  elif FLAGS.preset == 'economy':
    num_ensemble = 1
    recycle_tol  = 0.1
    max_extra_msa = 512
    max_msa_clusters = 256
  elif FLAGS.preset in ['super', 'super2']:
    num_ensemble = 1
    max_recycles = 20
    recycle_tol  = 0.1
    max_extra_msa = 1024
    max_msa_clusters = 512
    if FLAGS.preset == 'super2': num_ensemble = 2
  elif FLAGS.preset in ['genome', 'genome2']:
    num_ensemble = 1
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

  model_runners = {}
  for model_name in FLAGS.model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = max_recycles
    model_config.model.num_recycle = max_recycles
    model_config.model.recycle_tol = recycle_tol
    model_config.model.save_recycled = FLAGS.save_recycled

    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each target.
  for target in target_lst:
    target_name  = target['name']
    target_split = target['split']
    print(f"Info: working on target {target_name}")
    predict_structure(
        target=target,
        output_dir_base=FLAGS.output_dir,
        feature_dir_base=FLAGS.feature_dir,
        model_runners=model_runners,
        random_seed=random_seed,
        max_msa_clusters=max_msa_clusters,
        max_extra_msa=max_extra_msa,
        max_recycles=max_recycles,
        num_ensemble=num_ensemble,
        preset=FLAGS.preset,
        write_complex_features=FLAGS.write_complex_features,
        template_mode=FLAGS.template_mode
        )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'target_lst_path',
      'output_dir',
      'feature_dir',
      'model_names',
      'data_dir',
      'preset',
  ])

  app.run(main)
