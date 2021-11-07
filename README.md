# AF2Complex
## Predicting direct protein-protein interactions with deep learning

This is an extension of [AlphaFold 2](https://github.com/deepmind/alphafold) generalized for 
predicting strucural models of a protein complex. It is based on AlphaFold [version v2.0.1](https://github.com/deepmind/alphafold/releases/tag/v2.0.1) 
released by DeepMind in July 2021. We added a few features useful for modeling protein complexes 
that are not designed for but hidden in AF2's original release.

![Overview](image/af2complex_overview.jpg)

## Features

- Predicting structural models of a protein complex, e.g., a homooligomer or heterooligomer.
- No paired MSAs are not required for complex modeling
- New metrics for evaluating structural models of protein-protein interfaces
- Option to save the intermediate models during recycles
- Adding `genome`, `super`, `economy` presets
- Modularized workflow including data pipeline (stage 1), DL model 
inference (stage 2a) and MD minimization (stage 2b).

## Installation

If you have installed AlphaFold version 2.0.1, no additional package is required. If not,
please follow its official installation guide of 
[AlphaFold 2](https://github.com/deepmind/alphafold) first. This package requires the deep learning 
models with the capability of TM-score prediction (pTM). Please note that DeepMind's latest 
DL models re-trained for AlphaFold-multimer have not been tested and are not required.

After you have set up AlphaFold 2, please clone this repository and follow the guild provided
under the "example" folder.

## Example

Under the "example" directory, these are two CASP14 assembly targets as the examples. 
The goal is to predict the complex structures for these two targets,
one heterodimer (A1:B1) and one heterotetramer (A2:B2). The input features have been
generated for the individual protein sequences of these targets. We use these input
features of single chains to predict the structures of their complexe forms. Note that
the input features were generated using databases released before the start date
of CASP14.

Use `run_af_stage2a_comp.sh` shell script to run the examples. Be sure to modify it 
such that that the correct conda environment for AlphaFold is loaded and the correct 
path to the model parameters of AlphaFold 2 is pointed to.

```sh
./run_af_stage2a_comp.sh
```

For the purpose of evaluation, the experimental structures of these two examples are under
directory `ground_truth`.

### Input format
The component(s) of your target, be it a monomer or a complex, is (are) defined
in an input list file. The file is `test.lst` in the example above. The general format
is 

`A:2/B:2/C/D/D total_length target_name`

where the first column is the stoichiometry of the complex, using the names of the individual
sequences, total_length is the total number of amino acids of the putative complex, and
target_name is optional for naming the output subdirectory purpose.

### Input feature generation
If you need to generate feature inputs, check out the ```run_af_stage1.py```. This
script will generate input for individual protein sequences, which you may use
to predict single chain or multi-chain (assembly) structures.

### Output files
- `model_[1-5]_ptm_*.pdb`  Output structural (unreleaxed) models in the PDB format. 
   If there are mutliple chains, they are named alphabetically starting from 'A'
- `model_[1-5]_ptm_*.pkl`  Pickle file contains extra prediction from the DL model
- `ranking_all.json` or `model_*.json` Information about the model, such as predicted scores.
  By default, if it is complex predictioni, the models are ranked by the interface-score.
  For single chain, it is ranked by predicted TM-score.
- `features.pkl` This is a pickle file conatining features for structure prediction (stage 2a).
  It is the output from the data pipeline by running the stage 1 script. 
- `unrelaxed_model_*.pdb` Relaxed structural models by running the stage 2b script.

## Reference
- Predicting direct physical interactions in multimeric proteins with deep learning.
Mu Gao, Davi Nakajima An, Jerry M. Parks, and Jeffrey Skolnick. (2021)

- Highly accurate protein structure prediction with AlphaFold.
Jumper, J. et al., Nature 596, 583-589  (2021).

- ColabFold - Making protein folding accessible to all. Mirdita, M., Ovchinnikov, S. & Steinegger, M, bioRxiv, 2021.2008.2015.456425  (2021).

## Data sets
Data sets and predicted structural models described in the AF2Complex reference are 
available at [the CSSB website at Georgia Tech](https://sites.gatech.edu/cssb/af2complex/).

## Licencse

The source code is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of [the License](https://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Please follow the [license agreement](https://github.com/deepmind/alphafold#model-parameters-license) by DeepMind for using their neural network models of AlphaFold 2.
