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
# Modified to add predicted interface TM-score and Interface-score
# Metrics for complex model evaluation
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology
#
"""Functions for processing confidence metrics."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.special
import scipy.spatial

import time

def compute_plddt(logits: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = scipy.special.softmax(logits, axis=-1)
  predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
  return predicted_lddt_ca * 100


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = (breaks[1] - breaks[0])

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]],
                               axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  bin_centers = _calculate_bin_centers(alignment_confidence_breaks)

  # Tuple of expected aligned distance error and max possible error.
  return (np.sum(aligned_distance_error_probs * bin_centers, axis=-1),
          np.asarray(bin_centers[-1]))


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  aligned_confidence_probs = scipy.special.softmax(
      logits,
      axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(
          alignment_confidence_breaks=breaks,
          aligned_distance_error_probs=aligned_confidence_probs))
  return {
      'aligned_confidence_probs': aligned_confidence_probs,
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }


def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    is_probs: Optional[bool] = False,
    chain_mask: Optional[np.ndarray] = None,
    inter_chain_mask: Optional[np.ndarray] = None) -> np.ndarray:
  """Computes predicted TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.

  Returns:
    ptm_score: the predicted TM alignment score.
  """
  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])

  bin_centers = _calculate_bin_centers(breaks)

  num_res = np.sum(residue_weights)
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = max(num_res, 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
  # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  # Yang & Skolnick "Scoring function for automated
  # assessment of protein structure template quality" 2004
  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  # this may help interface prediction
  if d0 < 0.5: d0 = 0.02*num_res

  # Convert logits to probs
  if not is_probs:
    probs = scipy.special.softmax(logits, axis=-1)
  else:
    probs = logits

  # TM-Score term for every bin
  tm_per_bin = 1. / (1 + np.square(bin_centers) / np.square(d0))
  # E_distances tm(distance)
  predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

  # for interface-score
  if inter_chain_mask is not None:
      predicted_tm_term = predicted_tm_term * inter_chain_mask

  normed_residue_mask = residue_weights / (1e-8 + residue_weights.sum())

  if chain_mask is not None:
      per_alignment = np.sum(predicted_tm_term * normed_residue_mask * chain_mask, axis=-1)
      return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
  else:
      per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
      return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])


def predicted_interface_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_indices: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[int] = 4.5,
    is_probs: Optional[bool] = False,
    inter_chain_mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:

  """Computes predicted interfacial TM-score using only the residues
    that make up the interface between different chains in a protein
    complex (piTM)

    Score defined in the AF2Complex manuscript (2021)

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_indices: [num_res] index of each residue
    pos: [num_res, atom_type_num, 3] the predicted atom positions
    atom_mask: mask for atoms (each residue type has different number of atoms)
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface
    is_probs: boolean indicating whether the logits argument are probabilities

  Returns:
    None if target is a single chain, otherwise
    pitm_dict: dict of np.ndarrays containing
      "score" - piTM score for the sequence
      "num_residues" - number of residues in the interface
      "num_contacts" - number of contacts along the interface of protein complex

  """
  # only calculate piTMS if a long gap detected, suggesting multi-chain target
  if not residue_indices[-1]-residue_indices[0]-len(residue_indices) > 100:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  # Determine which chain each residue belongs to
  prev_id = np.roll(residue_indices, 1)
  diff = np.abs(residue_indices - prev_id)
  chain_breaks = diff > 100
  residue_chain_id = np.cumsum(chain_breaks)

  # calculates the minimum distance between each residue's heavy atoms
  def get_min_pairwise_dist(a, b, mask_a, mask_b):
    a = a[mask_a > 0.5]
    b = b[mask_b > 0.5]
    pairwise_dist = scipy.spatial.distance.cdist(a, b, metric='euclidean')
    return pairwise_dist.min()

  residue_mask = np.zeros(logits.shape[0]).astype(bool)
  contact_mask = np.zeros(logits.shape[:2]).astype(bool)

  for i in range(pos.shape[0]):
      for j in range(i+1, pos.shape[0]):  # use symmetry
          if residue_chain_id[i] != residue_chain_id[j]:
              if atom_mask[i].sum() == 0 or atom_mask[j].sum() == 0:
                  continue
              dist = get_min_pairwise_dist(pos[i], pos[j], atom_mask[i], atom_mask[j])
              residue_mask[i] = residue_mask[i] or dist < distance_threshold
              residue_mask[j] = residue_mask[j] or dist < distance_threshold
              contact_mask[i, j] = contact_mask[i, j] or dist < distance_threshold

  # return 0.0 if no inter-chain contacts found
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  # select only interfacial residues
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights = residue_weights * residue_mask

  # return the  piTM score and other interface data
  return {
    'score': predicted_tm_score(logits, breaks, residue_weights,
        is_probs=is_probs, inter_chain_mask=inter_chain_mask),
    'num_residues': np.asarray( residue_mask.sum() ),
    'num_contacts': np.asarray( contact_mask.sum() ),
  }


def interface_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_indices: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[int] = 4.5,
    is_probs: Optional[bool] = False) -> Dict[str, Union[np.ndarray, int]]:
  """Computes the interface-score of a complex model. This is a further
  tweak from piTM by looking the best tm-score estimate from other chains

    Score defined in the AF2Complex manuscript (2021)

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_indices: [num_res] index of each residue
    pos: [num_res, atom_type_num, 3] the predicted atom positions
    atom_mask: mask for atoms (each residue type has different number of atoms)
    distogram_logits: [num_res, num_res, 64] the logits for the distance bins \
      between residues.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface
  Returns:
    None if target is a single chain, otherwise
    pitm_dict: dict of np.ndarrays containing
      "score" - piTM score for the sequence
      "num_residues" - number of residues in the interface
      "num_contacts" - number of contacts along the interface of protein complex
  """
   # only calculate the score if a long gap detected, suggesting multi-chain target
  if not residue_indices[-1]-residue_indices[0]-len(residue_indices) > 100:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  # Determine which chain each residue belongs to
  prev_id = np.roll(residue_indices, 1)
  diff = np.abs(residue_indices - prev_id)
  chain_breaks = diff > 100
  residue_chain_id = np.cumsum(chain_breaks)

  # calculates the minimum distance between each residue's heavy atoms
  def get_min_pairwise_dist(a, b, mask_a, mask_b):
    a = a[mask_a > 0.5]
    b = b[mask_b > 0.5]
    pairwise_dist = scipy.spatial.distance.cdist(a, b, metric='euclidean')
    return pairwise_dist.min()

  residue_mask = np.zeros(logits.shape[0]).astype(bool)
  contact_mask = np.zeros(logits.shape[:2]).astype(bool)

  for i in range(pos.shape[0]):
      for j in range(i+1, pos.shape[0]):  # use symmetry
          if residue_chain_id[i] != residue_chain_id[j]:
              if atom_mask[i].sum() == 0 or atom_mask[j].sum() == 0:
                  continue
              dist = get_min_pairwise_dist(pos[i], pos[j], atom_mask[i], atom_mask[j])
              residue_mask[i] = residue_mask[i] or dist < distance_threshold
              residue_mask[j] = residue_mask[j] or dist < distance_threshold
              contact_mask[i, j] = contact_mask[i, j] or dist < distance_threshold

  # return 0.0 if no inter-chain contacts found
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  # select only interfacial residues
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights = residue_weights * residue_mask

  # create an inter-chain mask, where only residues relevant to tm scoring are ones.
  # note that residue pairs of the same chain subject to tm-scoring are zeros.
  # this force to select the best local frame from residues of other chains.
  next_id = np.roll(residue_indices, -1)
  diff = np.abs(next_id - residue_indices)
  termini = np.where( diff > 1 )[0] + 1

  score = 0
  num_chains = len(termini)
  full_length = len(residue_indices)
  # calculate the tm-score for interface residues of each chain, and sum them.
  for i in range(num_chains):
    k = termini[i]
    if i == 0: k_ = 0
    else: k_ = termini[i-1]
    inter_chain_mask = np.zeros((full_length, full_length))
    inter_chain_mask[k_:k,:] = 1
    inter_chain_mask[:,k_:k] = 1
    inter_chain_mask[k_:k,k_:k] = 0

    chain_mask = np.zeros(full_length, dtype=int)
    chain_mask[k_:k] = 1

    sc = predicted_tm_score( logits, breaks, residue_weights,is_probs=is_probs,
        chain_mask=chain_mask, inter_chain_mask=inter_chain_mask )
    #print(f'sc = {sc:.3f}')
    score += sc

  # return the  interaction score and other interface data
  return { 'score': score,
    'num_residues': np.asarray( residue_mask.sum() ),
    'num_contacts': np.asarray( contact_mask.sum() ) }
