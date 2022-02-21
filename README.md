<img src="image/af2complex_logo_hi.jpg" alt="AF2Complex Overview" width="800"/>

## Predicting and modeling protein complexes with deep learning
Accurate descriptions of protein-protein interactions are essential for understanding biological systems. Can we predict protein-protein interactions given an arbitrary pair of protein sequences, and more generally, can it identify higher order protein complexes? AF2Complex is born to address this question by taking advantage of [AlphaFold](https://github.com/deepmind/alphafold), sophisticated neural network models originally designed for predicting structural models of single protein sequences by DeepMind. Here, we extended it not only to model protein-protein interactions, but also use the confidence of the predicted structural models to predict possible interactions among multiple proteins, including transient interactions that are difficult to capture experimentally.

AF2Complex is an enhanced version of AlphaFold with many features useful for real-world application scenarios, especially for the prediction of a protein complex, either on a personal computer or at a supercomputer. Its initial development is based on AlphaFold [version v2.0.1](https://github.com/deepmind/alphafold/releases/tag/v2.0.1),
released by DeepMind in July 2021. After DeepMind released AlphaFold-Multimer [version v2.1.1](https://github.com/deepmind/alphafold/releases/tag/v2.1.1) in November 2021, AF2Complex has been updated to support the multimer deep learning models of AlphaFold. Details of our development, including large-scale  performance evaluations and exemplary applications, have been described in [this work](https://www.biorxiv.org/content/10.1101/2021.11.09.467949v1).

## Updates and Features

#### Version 1.2 (2022-02-19)

- Added support to AF-Multimer neural network models in both paired and unpaired MSAs modes
- Domain cropping pre-generated full monomer features (in unpaired MSAs mode only)
- Checkpoint option for model inference
- Refactored code

#### Version 1.0 (2021-11-09)

- Predicting structural models of a protein complex
- Paired MSAs not required for complex modeling
- Metrics for evaluating structural models of protein-protein interfaces
- Option to save the intermediate models during recycles
- Added `genome`, `super`, `economy` presets
- Modularized workflow including feature generation, DL model inference and MD minimization

## Installation

This package has identical software dependency and hardware requirement as AlphaFold
[version v2.1.1](https://github.com/deepmind/alphafold/releases/tag/v2.1.1).
If you have installed AlphaFold version v2.1.1, no additional software installation other than this one is required. If not, please follow the official installation guide of [AlphaFold 2](https://github.com/deepmind/alphafold) first. Note that if you have already generated input features, or just evaluating this packages by following the examples we provided, you do *not* need to install any sequence library or third-party sequence searching tools. After resolving all python dependency required by AlphaFold, you are (almost) good to run the examples below.

The other items you need are the deep neural network models trained by DeepMind for AlphaFold. Running this package requires the DL models with the TM-score prediction capability (i.e, monomer_pTM or multimer models). In AF's releases, these models are named as `model_x_ptm.npz` or `model_x_multimer.npz`. Note that AlphaFold-multimer's models (version 2.1.1) are not required if you only wish to run the original monomer DL models. The installation of AlphaFold 2 environment could take hours or a couple of days, dependent on whether you choose to do a full installation including all sequence libraries.

After you have set up AlphaFold 2, simply clone this repository by
```sh
git clone https://github.com/FreshAirTonight/af2complex
```
 and follow the guide below to run the demo examples. The installation of this package itself takes only seconds or several minutes, dependent on your network speed.

## Example

Under the "example" directory, there are two CASP14 multimeric targets, H1065 and H1072, as the examples. The goal is to predict the complex structures for these two targets, one heterodimer (A1:B1) and one heterotetramer (A2:B2). The input features have been generated for them and placed under the subdirectory `af_fea`. We use these input features to predict the structures of their complexes forms. Note that the input features were generated using databases released before the start date of CASP14.

Use `run_af2comp.sh` shell script to run the examples. Be sure to modify it
such that that the correct conda environment for AlphaFold is loaded and the correct path to the model parameters of AlphaFold 2 is pointed to.

```sh
./run_af2comp.sh
```

The output structural models are under the subdirectory `af2c_mod`. By default, this run uses the `multimer_np` model preset, which assembles input monomer features using unpaired MSAs and applys the `multimer_x_multimer` models for inference. You may also test the `monomer_ptm` model preset, which calls the original monomer DL modles on unpaired MSA feature sets. For the purpose of comparison, the experimental structures of these two examples are provided under subdirectory `ground_truth`.

The run time for a single model prediction is about two to five minutes on a workstation equipped with an Nvidia RTX6000 GPU. It is recommended to run these examples on a machine with a modern GPU such as Nvidia RTX6000, P100, V100, or higher. We have not tested the code on a CPU-only computer, but running these examples should not be a problem on a modern computer.

## Feature generation
For your convenience, we also provide the pre-generated features (both monomer and complex feature sets) for our benchmark data sets, and for the E. coli proteome (~4,400 proteins) at [Zenodo](https://doi.org/10.5281/zenodo.6084186).

If you apply this package to a new target. The first step is to generate input features. For model presets using unpaired MSAs, i.e., `monomer_ptm` or `multimer_np`, what you need is the input features of each individual (unique monomer) protein sequences of your complex target. For each monomer, its features are packed in a `features.pkl` file, generated by AF2's monomer data pipeline. Alternatively, if you would like to use the paired MSAs by DeepMind's multimer data pipeline, you may do so and generate the features for a complex target directly. The multimer features can only be used in comibnation with the "multimer" model preset. Certain features, such as domain cropping, are not supported in this model preset.

For the purpose of efficient computing, we have created a staged AF2 workflow and provide a script ```run_af2c_fea.py``` for the stage of feature generation. This script will output features (in python pickle format) for an individual protein sequence from the monomer data pipeline of AF, or for a whole complex target if you choose to use DeepMind's multimer data pipeline. Both these types of features can be fed into the next stage of AF2Complex in model inference. However, if you would like to use the unpaired MSAs (either `monomer_ptm` or `multimer_np` model preset), only the features of monomers are needed. This is designed to efficiently re-use the input features of monomers that one needs to generate one time.

## Target syntax

After collecting the input features, you may use them to predict a complex structure, using the script ```run_af2c_mod.py```, which runs through the deep learning model inference step. The stoichiometry of your target, be it a monomer or a complex, is defined in an input list file. In the example we provided, the target list file is named `test.lst`. The general format of a target is as the follows,

`A:2/B:2/C/D/E <total_length> <target_name>`

where the first column defines the stoichiometry of the complex, e.g., `A:2/B:2/C/D/E`, using the IDs of the individual sequences, `:<num>` after each protein defines its homo copy number, and `/` to separate distinct monomers. The second column, `<total_length>`, is the total number of amino acids of the putative complex. This is mainly for load-balance in a large-scale cluster run, parsed but not used by the model inference python script. The third column, `<target_name>`, is the name of the output sub-directory.

In the example target above, the complex is made of five protein sequences named A to E, and protein A and B each have two copies. During a prediction, the program will look for individual input features of A to E under the input feature directory, e.g, `$inp_dir/A/features.pkl`, and then assemble them into the features for complex structure prediction. If you provide only a single protein without a copy number, e.g., `A <seq_length>`, it reverts to a structural prediction of a single protein A.

A more advanced example of using domain cropping feature is like the follows

`A|19-200;500-700:2/B/C 1788 A2BC`

where the residue ranges, 19 to 200 and 500 to 700, are taken out from A's full length input features for modeling A2BC, composed of two copies of A, single copy of B and C, and with a total size of 1788 AAs. This format allows convenient and rapid tests of part of a large sequence, and also avoid possible errors caused by using a partial sequence to derive MSAs.

## Checkpoint

To help tackle a large target on a limited computing resource, AF2Complex provides a checkpoint option in the model inference stage. An example usage is included in the example above. Note that the checkpoint is saved at the end of the all recycles done (due to a limitation imposed by JAX). Therefore, to use this feature, one has to estimate beforehand how many recycles will be done before saving a checkpoint. Additionally, for a large target, it is recommended to turn off intermediate recycle output (by setting `save_recycled=0`) to save memory cost, and to use multiple short runs with the checkpoint option to get the intermediate recycle structures, e.g., every two, or three recycles.

## Model relaxation

Optionally, you may run a MD minimization to reduce clashes (if exist) in "un-relaxed" models generated above to obtain "relaxed" models. The script for this purpose is ```run_af2c_min.py```. And a demo scrit is provided under the `example` directory. To test it after you have successfully run the example above,
```sh
./run_relaxation.sh
```
This script will launch the relaxation protocol on all un-relaxed structural models generated by running `run_af2comp.sh`. Note that the relaxation is a molecular dynamics minimization procedure, which cannot eliminate severe clashes as observed in some models of large complexes generated with the multimer DL models.

## Output files

- `model_[1-5]_ptm_*.pdb` and `model_[1-5]_multimer_*.pdb`  Output structural (unreleaxed) models in the PDB format. The naming scheme is DL model followed by the date and the six digit random seed employed. If there are multiple chains in a PDB file, they are named alphabetically starting from 'A'.
- `model_[1-5]_ptm_*.pkl` and `model_[1-5]_multimer_*.pkl` Pickle file contains extra information from the DL model.
- `ranking_all.json` or `model_*.json` Information about the model, such as predicted scores.By default, if there are multiple complex model predictions, the models are ranked bytheir interface-scores. For single chain models or multipel chain models without interaction, they are ranked by predicted TM-score.
- `features.pkl` This is a pickle file containing features generated from the data pipeline
  by running the `run_af2c_fea` script. It is employed for structure prediction during DL model inference.
- `relaxed_model_*.pdb` Relaxed structural models by running the `run_af2c_min` script.
- If you choose to save intermediate PDB files from recycling, the files will be under a subdirectory named `recycled`.
- If you choose to use checkpoint, a checkpoint will be under a subdirectory named `checkpoint`, in which a python pickle file named by `model_*_[checkpoint_tag].pkl` is saved for reloading in a future run.

## Reference

- Predicting direct physical interactions in multimeric proteins with deep learning.
Mu Gao, Davi Nakajima An, Jerry M. Parks, and Jeffrey Skolnick.
[bioRxiv, 2021.11.09.467949](https://doi.org/10.1101/2021.11.09.467949) (2021).

- Proteome-scale deployment of protein structure prediction workflows on the summit supercomputer.
Mu Gao, Mark Coletti, et.al., [arXiv, 2201.10024](https://arxiv.org/abs/2201.10024) (2022).

- Highly accurate protein structure prediction with AlphaFold.
Jumper, J. et al., Nature 596, 583-589  (2021).

- Protein complex prediction with AlphaFold-Multimer.
Evans, R. et al. bioRxiv, 2021.10.04.463034 (2021).

- ColabFold - Making protein folding accessible to all. Mirdita, M., et.al., bioRxiv, 2021.2008.2015.456425  (2021).

## Data sets and pre-generated input features

Benchmark data sets used for benchmarking AF2Complex and the top computational models of E. coli Ccm system I are available at [Zenodo](https://doi.org/10.5281/zenodo.6084186).

Pre-generated input features of the full *E. coli* proteome are also provided in the same depository. You are welcome to use these features to predict the interactions among any *E. coli* proteins of your choice, or the whole interactome! If you use the data set and code, please cite [our work](https://doi.org/10.1101/2021.11.09.467949).

## Acknowledgment

We thank all scientists who did decades of hard works on protein sequencing and structural modeling. They have laid the groundwork that enables the development of a deep learning approach.

We thank DeepMind for their open-sourced AlphaFold development.

This work was funded in part by the Office of Biological and Environmental Research of the Department of Energy, and the Division of General Medical Sciences of the National Institute Health. The research used resources supported in part by the Director’s Discretion Project at the Oak Ridge Leadership Computing Facility, and the Advanced Scientific Computing Research (ASCR) Leadership Computing Challenge (ALCC) program. We also acknowledge the computing resources provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology.

## Licencse

The source code is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of [the License](https://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Please follow the [license agreement](https://github.com/deepmind/alphafold#model-parameters-license) by DeepMind for using their neural network models of AlphaFold.
