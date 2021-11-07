"""Run MD minization to relax a protein structure model from AF2"""
import os
import pickle
import re
import time

from absl import app
from absl import flags
from tqdm import tqdm

from alphafold.relax  import relax
from alphafold.common import protein

from run_alphafold_stage2a_comp import _read_target_file, FLAGS

import numpy as np



MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

  # read list of targets
  target_lst = _read_target_file( FLAGS.target_lst_path )

  for target in target_lst:
    # get target name
    target_name  = target['name']
    target_name = re.sub(":", "_x", target_name)
    target_name = re.sub("/", "+", target_name)
    target_dir = os.path.join(FLAGS.output_dir, target_name)

    for afile in os.listdir(target_dir):
      # find all unrelaxed pdb files, relaxed ones with 'relaxed' as prefix
      if ".pdb" in afile and afile.startswith("model_"):
        print(f"Info: {target_name} processing {afile}")
        unrelaxed_pdb_file = afile
        unrelaxed_pdb_path = os.path.join(target_dir, unrelaxed_pdb_file)
        with open(unrelaxed_pdb_path, "r") as f:
            unrelaxed_pdb_str = f.read()

        unrelaxed_protein = protein.from_pdb_string( unrelaxed_pdb_str )

        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        relaxation_time = time.time() - t_0

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(target_dir, f'relaxed_{unrelaxed_pdb_file}')
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb_str)

        print(f"Info: {target_name} relaxation done, time spent {relaxation_time:.1f} seconds")


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'target_lst_path',
      'output_dir',
      'feature_dir',
      'template_mode',
  ])

  app.run(main)
