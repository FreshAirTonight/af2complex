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
# Modified for complex structure prediction
# Mu Gao and Davi Nakajima An
#

"""Code for constructing the model."""
import os
from typing import Any, Mapping, Optional, Union, List
import pickle

from absl import logging
from alphafold.common import confidence
from alphafold.model import features
from alphafold.model import modules
from alphafold.model import modules_multimer
import haiku as hk
import jax
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
import tree


def get_confidence_metrics(
    prediction_result: Mapping[str, Any],
    multimer_mode: bool,
    residue_index: Optional[np.ndarray] = None,
    edge_contacts_thres: Optional[int] = 10,
    superid2chainids: Optional[Mapping[int, List[int]]]=None) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""
  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        asym_id=None)
    if multimer_mode:
      # Compute the ipTM only for the multimer model.
      confidence_metrics['iptm'] = confidence.predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          interface=True)
      confidence_metrics['iptm+ptm'] = (
          0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

    # must convert jax array to np array, otherwise some interplay between
    # jax array and the loops in the itms function dramatically slowdowns the speed
    confidence_metrics['pitm'] = confidence.predicted_interface_tm_score(
        np.asarray(prediction_result['predicted_aligned_error']['logits']),
        np.asarray(prediction_result['predicted_aligned_error']['breaks']),
        np.asarray(prediction_result['structure_module']['final_atom_positions']),
        np.asarray(prediction_result['structure_module']['final_atom_mask']),
        np.asarray(prediction_result['predicted_aligned_error']['asym_id']),
        )
    
    confidence_metrics['interface'] = confidence.interface_score(
        np.asarray(prediction_result['predicted_aligned_error']['logits']),
        np.asarray(prediction_result['predicted_aligned_error']['breaks']),
        np.asarray(prediction_result['structure_module']['final_atom_positions']),
        np.asarray(prediction_result['structure_module']['final_atom_mask']),
        np.asarray(prediction_result['predicted_aligned_error']['asym_id']),
        )
    asym_id = np.asarray(prediction_result['predicted_aligned_error']['asym_id'])
    if not np.all(asym_id == asym_id[0]): # multi-chain target
      confidence_metrics['cluster_analysis'] = confidence.cluster_analysis(
        np.asarray(prediction_result['predicted_aligned_error']['asym_id']),
        np.asarray(prediction_result['structure_module']['final_atom_positions']),
        np.asarray(prediction_result['structure_module']['final_atom_mask']),
        edge_contacts_thres=edge_contacts_thres,
        superid2chainids=superid2chainids,
      )

  #if not multimer_mode:
    # Monomer models use mean pLDDT for model ranking.
    #confidence_metrics['ranking_confidence'] = np.mean(
        #confidence_metrics['plddt'])
  return confidence_metrics


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
               is_training=False):
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode

    if self.multimer_mode:
      def _forward_fn(batch, prev):
        model = modules_multimer.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=is_training,
            return_representations=True,
            prev=prev)
    else:
      def _forward_fn(batch, prev):
        model = modules.AlphaFold(self.config.model)
        return model(
            batch,
            is_training=is_training,
            compute_loss=False,
            ensemble_representations=True,
            return_representations=True,
            prev=prev)

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init = jax.jit(hk.transform(_forward_fn).init)


  def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
    """Initializes the model parameters.

    If none were provided when this class was instantiated then the parameters
    are randomly initialized.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: A random seed to use to initialize the parameters if none
        were set when this class was initialized.
    """
    if not self.params:
      # Init params randomly.
      rng = jax.random.PRNGKey(random_seed)
      self.params = hk.data_structures.to_mutable_dict(
          self.init(rng, feat))
      logging.warning('Initialized parameters randomly')

  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:
    """Processes features to prepare for feeding them into the model.

    Args:
      raw_features: The output of the data pipeline either as a dict of NumPy
        arrays or as a tf.train.Example.
      random_seed: The random seed to use when processing the features.

    Returns:
      A dict of NumPy feature arrays suitable for feeding into the model.
    """
    if self.multimer_mode:
      return raw_features

    # Single-chain mode.
    if isinstance(raw_features, dict):
      return features.np_example_to_features(
          np_example=raw_features,
          config=self.config,
          random_seed=random_seed)
    else:
      return features.tf_example_to_features(
          tf_example=raw_features,
          config=self.config,
          random_seed=random_seed)

  def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
    self.init_params(feat)
    logging.info('Running eval_shape with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))
    shape = jax.eval_shape(self.apply, self.params, jax.random.PRNGKey(0), feat)
    logging.info('Output shape was %s', shape)
    return shape

  def predict(self, feat: features.FeatureDict, random_seed: int,
            prev=None, prev_ckpt_iter=0, asym_id_list=None, asym_id=None, 
            edge_contacts_thres=10) -> Mapping[str, Any]:
    """Makes a prediction by inferencing the model on the provided features.

    Args:
      feat: A dictionary of NumPy feature arrays as output by
        RunModel.process_features.
      random_seed: The random seed to use when running the model. In the
        multimer model this controls the MSA sampling.

    Returns:
      A dictionary of model outputs.
    """
    self.init_params(feat)
    logging.info('Running predict with shape(feat) = %s',
                 tree.map_structure(lambda x: x.shape, feat))

    feat = {k: v for k, v in feat.items() if v.dtype != 'O'}
    result, recycles = self.apply(
      self.params, jax.random.PRNGKey(random_seed), feat, prev=prev)

    if self.checkpoint_file is not None:
      if os.path.exists(self.checkpoint_file):
          # rename previous checkpoint file before writing a new checkpoint
          prev_ckpt_file = self.checkpoint_file + f".{prev_ckpt_iter:02d}"
          os.rename(self.checkpoint_file, prev_ckpt_file)
      else:
          # create a checkpoint for the first time, check path
          ckpt_dir, ckpt_file = os.path.split(self.checkpoint_file)
          if not os.path.exists(ckpt_dir):
              os.makedirs(ckpt_dir)

      if prev_ckpt_iter: prev_ckpt_iter += 1
      prev_iter = prev_ckpt_iter + recycles[0]

      logging.info(f"Saving checkpoint at recycle round {prev_iter}")
      with open(self.checkpoint_file, "wb") as f:
        pickle.dump(
          ({
            'prev_pos':
              result['structure_module']['final_atom_positions'],
            'prev_msa_first_row': result['representations']['msa_first_row'],
            'prev_pair': result['representations']['pair'],
          }, prev_iter), f, protocol=4)
        f.close()
        logging.info(f"Checkpoint saved to {self.checkpoint_file}")

    del result['representations'] # save space

    # This block is to ensure benchmark timings are accurate. Some blocking is
    # already happening when computing get_confidence_metrics, and this ensures
    # all outputs are blocked on.
    #jax.tree_map(lambda x: x.block_until_ready(), result)
    if self.multimer_mode:
        res_index = feat['residue_index']
    else:
        res_index = feat['residue_index'][0]
      
    # join superchains under one asym_id (if needed)
    asym_id, superid2chainids = confidence.join_superchains_asym_id(asym_id, asym_id_list)
    result['predicted_aligned_error']['asym_id'] = asym_id

    result.update(
      get_confidence_metrics(
        result, multimer_mode=self.multimer_mode, residue_index=res_index, superid2chainids=superid2chainids))
    #logging.info('Output shape was %s',
    #             tree.map_structure(lambda x: x.shape, result))

    if self.config.model.save_recycled:
      *_, recycled_info = recycles
     # must convert jax array to np array, otherwise some interplay between
     # jax array and the loops in the AF2Complex metric functions dramatically slow downs the calculations     
      structs = np.asarray(recycled_info['atom_positions'])
      structs_masks = np.asarray(recycled_info['atom_mask'])
      plddt = np.asarray(recycled_info['plddt'])
      palign_logits = np.asarray(recycled_info['pred_aligned_error_logits'])
      palign_break = np.asarray(recycled_info['pred_aligned_error_breaks'])
      tol_values = np.asarray(recycled_info['tol_values'])
      recycled_info_ = []

      for i, (s, m, p, a_logits, a_break, tol_val) in enumerate(zip(
        structs, structs_masks, plddt, palign_logits, palign_break, tol_values
      )):
        r = {
          "structure_module": {
            "final_atom_positions": s,
            "final_atom_mask": m,
          },
          "plddt": confidence.compute_plddt(p),
          "ptm": confidence.predicted_tm_score(
            a_logits,
            a_break,
          ),
          "tol_val": tol_val,
          "pitm": confidence.predicted_interface_tm_score(
            a_logits,
            a_break,
            s,
            m,
            asym_id,
          ),
          "interface": confidence.interface_score(
            a_logits,
            a_break,
            s,
            m,
            asym_id,
          ),
        }

        recycled_info_.append(r)
      recycles = (*_, recycled_info_)
      recycled_info.clear()


    return result, recycles
