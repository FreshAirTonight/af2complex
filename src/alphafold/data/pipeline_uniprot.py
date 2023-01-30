# Modified by Mu Gao to consider only UniProt as the sequence database
#
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

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msa: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msa:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()

  if not msa:
    raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
  for sequence_index, sequence in enumerate(msa.sequences):
    if sequence in seen_sequences:
      continue
    seen_sequences.add(sequence)
    int_msa.append(
      [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
    deletion_matrix.append(msa.deletion_matrix[sequence_index])
    identifiers = msa_identifiers.get_identifiers(
      msa.descriptions[sequence_index])
    species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msa.sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               uniprot_database_path: str,
               template_searcher: TemplateSearcher,
               template_featurizer: templates.TemplateHitFeaturizer,
               uniprot_max_hits: int = 30000,
               use_precomputed_msas: bool = False,
               add_species: bool = False):
    """Initializes the data pipeline."""

    self.template_searcher = template_searcher
    self.template_featurizer = template_featurizer
    self.use_precomputed_msas = use_precomputed_msas
    self.add_species = add_species
    if add_species:
        self.uniprot_msa_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniprot_database_path)
        self.uniprot_max_hits = uniprot_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    uniprot_out_path = os.path.join(msa_output_dir, 'uniprot_hits.sto')
    jackhmmer_uniprot_result = run_msa_tool(
        msa_runner=self.uniprot_msa_runner,
        input_fasta_path=input_fasta_path,
        msa_out_path=uniprot_out_path,
        msa_format='sto',
        use_precomputed_msas=self.use_precomputed_msas,
        max_sto_sequences=self.uniprot_max_hits)

    msa_for_templates = jackhmmer_uniprot_result['sto']
    msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
    msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(
        msa_for_templates)

    uniprot_msa = parsers.parse_stockholm(jackhmmer_uniprot_result['sto'])

    if self.template_searcher.input_format == 'sto':
      pdb_templates_result = self.template_searcher.query(msa_for_templates)
    elif self.template_searcher.input_format == 'a3m':
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
      pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
    else:
      raise ValueError('Unrecognized template input format: '
                       f'{self.template_searcher.input_format}')

    pdb_hits_out_path = os.path.join(
        msa_output_dir, f'pdb_hits.{self.template_searcher.output_format}')
    with open(pdb_hits_out_path, 'w') as f:
      f.write(pdb_templates_result)

    pdb_template_hits = self.template_searcher.get_template_hits(
        output_string=pdb_templates_result, input_sequence=input_sequence)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        hits=pdb_template_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(uniprot_msa)
    #logging.info('UniProt MSA size: %d sequences.', len(uniprot_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
