"""Run MD minization to relax a protein structure model from AF2"""
# This is a modification of DeepMind's run_alphafold.py
#
# Input: un-relaxed protein models in the PDB format
# Output: energy-minimized protein models in the PDB format
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology
#
import os
import pickle
import re
import time
import json

from absl import app
from absl import flags
from absl import logging
#from tqdm import tqdm

from alphafold.relax  import relax
from alphafold.common import protein
from alphafold.data.complex import read_af2c_target_file

import numpy as np

logging.set_verbosity(logging.INFO)

flags.DEFINE_string('target_lst_path', None, 'Path to a file containing a list of targets '
                  'in any monomer, homo- or hetero-oligomers '
                  'configurations. For example, TarA is a monomer, TarA:2 is a dimer '
                  'of two TarAs. TarA:2/TarB is a trimer of two TarA and one TarB, etc.'
                  )
flags.DEFINE_string('input_dir', None, 'Path to a directory that '
                    'contains unrelaxed models in the PDB format.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the relaxed models.')
flags.DEFINE_string('model_str', 'model_', 'Only relax a model with a specified '
                    'string, e.g., ranked_top1. The default will process all model_* pdb files')
flags.DEFINE_boolean('use_gpu_relax', None, 'Whether to relax on GPU. '
                     'Relax on GPU can be much faster than CPU, so it is '
                     'recommended to enable if possible. GPUs must be available'
                     ' if this setting is enabled.')


FLAGS = flags.FLAGS

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    use_gpu=FLAGS.use_gpu_relax)

  if FLAGS.output_dir is None:
      FLAGS.output_dir = FLAGS.input_dir

  # read list of targets
  target_lst = read_af2c_target_file( FLAGS.target_lst_path )

  for target in target_lst:
    # get the name of a target
    target_name  = target['name']

    input_dir = os.path.join(FLAGS.input_dir, target_name)
    output_dir = os.path.join(FLAGS.output_dir, target_name)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  
    relax_metrics = {}
    for afile in os.listdir(input_dir):
      # find all unrelaxed pdb files, ignore ones with 'relaxed' as prefix
      if not afile.endswith(".pdb") or afile.startswith("relaxed_"):
          continue

      if FLAGS.model_str in afile:
        logging.info(f"{target_name} processing {afile}")
        unrelaxed_pdb_file = afile
        unrelaxed_pdb_path = os.path.join(input_dir, unrelaxed_pdb_file)
        with open(unrelaxed_pdb_path, "r") as f:
            unrelaxed_pdb_str = f.read()

        unrelaxed_protein = protein.from_pdb_string( unrelaxed_pdb_str )

        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, log, _ = amber_relaxer.process(prot=unrelaxed_protein)
        relaxed_pdb_str, _, violations = amber_relaxer.process(prot=unrelaxed_protein)
        relax_metrics[f'relaxed_{unrelaxed_pdb_file}'] = {
            'remaining_violations': violations,
            'remaining_violations_count': sum(violations)
        }       
        relaxation_time = time.time() - t_0

        # Fix residue index, and ignore hydrogen atoms
        relaxed_protein = protein.from_relaxation( relaxed_pdb_str, unrelaxed_protein.residue_index )
        relaxed_pdb_str = protein.to_pdb(relaxed_protein)

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(output_dir, f'relaxed_{unrelaxed_pdb_file}')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)

        logging.info(f"{target_name} relaxation done, time spent {relaxation_time:.1f} seconds, "
            f"Efinal {log['final_energy']:.2f}, rmsd {log['rmsd']:.2f}")

    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'target_lst_path',
      'input_dir',
      #'use_gpu_relax',
  ])

  app.run(main)
