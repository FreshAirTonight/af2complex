## Examples of applying AF2Complex

We show how to run AF2Complex to predict the structures of three multimeric protein assembly targets from [CASP14](https://predictioncenter.org/casp14/targetlist.cgi) as examples. The examples cover both the monomeric and multimeric deep learning models released by DeepMind, different MSA pairing modes, and the interface score evaluation.

It is recommended to use a computer with a modern GPU (e.g., Nvidia RTX, V100, A100)
to test these examples because of fast result turnaround within minutes, except for Example 4 (which takes about 35 minutes on an A100 machine). But with patience you should be able to complete these examples on a modern CPU computer as well.

Before running these examples, please ensure that correct environment for AF2Complex has been properly set up. To do so, use a text editor to open an example shell script, and modify it accordingly. As instructed in the script, you must provide the correct conda environment of AlphaFold, if not already loaded; and provide the correct path to the weight parameter files of [AlphaFold 2 neural network models](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar). In AF's releases, these model parameters are named as `params_model_x_ptm.npz` (AF version 2.0.x), `params_model_x_multimer.npz` (AF version 2.1.x), `params_model_x_multimer_v2.npz` (AF version 2.2.x), and `params_model_x_multimer_v3.npz` (AF version 2.3.x). In the examples below, we employ two sets of parameters released in AF v2.0 and v2.3.

### Example 1
Predicting the structure of a heterodimer (H1065) using two AF-Multimer DL models. In this example, the target is composed of two monomers (T1065s1 and T1065s2). AF2Complex first retrieves the input features of each monomer under the sub-directory `af2c_fea`, then pairs their MSAs using the MSA pairing option `all`. The example has two files,

- `example1.sh` The shell script to run AF2Complex model inference.
- `targets/example1.lst` The list file that defines the stoichiometry of the target.

Run the script with the command,
```sh
./example1.sh
```

This run uses two `multimer_v3` models to make complex structure prediction. The output structural models (in PDB format) are under the subdirectory `af2c_mod/H1065`. For the purpose of comparison, the experimental structures of these two examples are provided under subdirectory `ground_truth`.
You may also view the experimental structure of this target directly in the
[Protein Data Bank](https://www.rcsb.org/structure/7m5f).

To view your predicted structures, you may take advantage of the online molecular viewer Mol* [in a simple](https://www.rcsb.org/3d-view) or [more advanced version](https://molstar.org/viewer/) by uploading your predicted structure coordinate files (.pdb). The other two types of output files are pickle files (.pkl), which contain detailed results of the prediction; and json files (.json), which contain statistics of the prediction. Example output is under `af2c_mod_examples/H1065`. Note the pickle files are not included in the example output because they are too large for github sharing.

### Example 2
Predicting the structure of a tetramer target (H1072) using two AF2 monomeric DL models
in the unpaired MSA mode. The target has two monomers (T1072s1 and T1072s2), each contribute two copies to make the tetramer. The stoichiometry is A2:B2. The example has two files,

- `example2.sh` The shell script to run AF2Complex model inference.
- `targets/example2.lst` The list file that defines the target, such as stoichiometry and copy numbers in this case.

Run the script with the command,
```sh
./example2.sh
```

This run uses two `monomer_ptm` models to make complex structure prediction. The output structural models (in PDB format) are under the subdirectory `af2c_mod/H1072`. Example output is under `af2c_mod_examples/H1072`. For the purpose of comparison, the experimental structures of these two examples are provided under subdirectory `ground_truth`. The experimental coordinates of this target are available in
the [PDB](https://www.rcsb.org/structure/6r17).

### Example 3
Predicting the structure of the above two targets (H1065 and H1072) simultaneously using one AF-Multimer v2 DL model in unpaired MSA mode. The example has two files,

- `example3.sh` The shell script to run AF2Complex model inference.
- `targets/example3.lst` The target list file. Note that `+` instead of `/` is used to separate two complexes. The `+` specify the interface the iScore will evaluate.

Run the script with the command,
```sh
./example3.sh
```

In this example, the interface score (iScore) evaluation is conducted between these two complexes, i.e., to assess whether there is any interaction between these two complexes. The resulting structure model comprises two protein complex clusters, one the dimer, and the other the tetramer.

### Example 4
Predicting the structure of a homo-dodecamer using one AF2 monomeric DL model
in the cyclic MSA paring mode. The target H1060v4 is an homo-oligomer, and the input monomer features are under `af2c_fea/H1060s3`. This example takes about half an hour using an A100 GPU. Eight recycles are recommended for this example. If you can only run short recycles, checkpoint option is activated such that you may continue the modeling in multiple runs. The example has two files,

- `example4.sh` The shell script to run AF2Complex model inference.
- `targets/example4.lst` The target list file called by the shell script

Run the script with the command,
```sh
./example4.sh
```
As of Aug 2022, the expreimental structure of this target have not been released. But it is known that the target structure is a [circular ring](https://predictioncenter.org/casp14/showpdbimage.cgi?target=H1060v4). The cyclic MSA mode provided by AF2Complex gives you a better chance to obtain such a ring-like structure, see an example output under `af2c_mod_examples/H1060v4`. It may take several tries to get it, and you may also see an ellipse structure or multiple rings. You may also try multimer v3 models with or without the cyclic MSA mode, or multimer v1 models in the unpaired MSA mode. Note that the radius of these circular rings resulted from different models may not be the same in this challenging example.

### A general model inference script
The general script for AF2Complex model inference is provided in the script
`run_af2comp.sh`, which processes targets specified in `test.lst`. The target list files defined multiple targets, each with its own stoichiometry, domain ranges, copy numbers, etc.

Run the script with the command,
```sh
./run_af2comp.sh
```

## Interface score calculation
You can calculate the interface score (iScore) using a result pickle file generated by AF2Complex model inference. In this example, we re-calculate the interface scores using the pickle files generated in running Example 2.

- `calc_interface_score.sh` The shell script to calculate the interface score.
- `targets/test_iscore.lst` The target list file called by the shell script.

It conducts two iScore evaluations as specified in `targets/test_iscore.lst` on each pickle file. The first iScore is to evaluate all intefaces of the full tetrameric complex. The second is to evaluate only the interactions between two homodimers. In the latter case, it uses a magic `+` to divide individual monomeric chains into two sets of superchains, each with multiple chains. When `+` is present in the stoichiometry of a target, iScore is calculated only on components separated by `+`, instead of all interfaces by default without using any `+`. This is useful for PPI screening purpose, e.g., screening a protein complex with multiple chains against arbitrary library entries, either monomers or known multimers. We consider predictions of medium, high, and very high confidence, progressively, at iScore cutoffs of 0.4, 0.5, 0.7.

## Generating input features
If you apply this package to a new target. The first step is to generate input features. `run_fea_gen.sh` is an example shell script to derive input features for subsequent model inference. To run it, you need to first install third-party sequence tools and libraries as required by AlphaFold 2. And modify this script to reflect correct installation path for sequence alignment accordingly.

These are the `feature_mode` to choose from:

- `monomer+species` Modified monomer data pipeline to include additionally species information. We recommend that you use this mode, which enables a flexibility using either unpaired MSAs or paired MSAs in various paring modes available in AF2Complex. The monomer features generated with these two monomer modes can be used to predict arbitrary combinations of the monomers for modeling a multimeric target.

- `monomer+fullpdb` In addition to add the species information, the template search uses the template pipeline employed for predicting multimers, instead of using the original monomer template pipeline. The advantage is that this template search covers the full PDB library.

- `monomer` AF v2.0 data pipeline to generate input features for predicting monomer structures. The features generated by this option are used by AF2Complex v1.2.2 and below.

- `multimer` AF v2.2+ data pipeline to generate input features for predicting multimer structures. The multimer features generated by this option can only be used in combination with the "multimer" model preset. It is provided such that you can run the official DeepMind's procedure. Certain AF2Complex features, such as domain cropping, are not supported in this model preset.

The input features used by examples above were all generated by using the option `monomer+species`.

## Model relaxation
An example shell script of model relaxation is provided in `run_relaxation.sh`.
To test it after you have successfully run the example above,

```sh
./run_relaxation.sh
```

This script will launch the relaxation protocol on all un-relaxed structural models generated by running the examples above. Note that the relaxation is a molecular dynamics minimization procedure, which cannot eliminate severe clashes as observed in some models of large complexes generated by deep learning models.

## Other files and directories

- `examples.sh` Run through Examples 1 to 3
- `targets/examples.lst`  Input targets and parameters called by `examples.sh`
- `af2c_fea`  Directory contains pre-generated features of monomers and complex
targets. For a monomer, the input features contains species information. See
feature generation above
- `af2c_mod_examples`  Examples of output models
- `af2c_mod` Output directory created by running the example scripts
- `ground_truth` Coordinates (PDB format) of target H1065 and H1072 determined by experiments
- `targets/test_custom_msa.lst` and `test.adj` An example of customized MSAs, an experimental feature.
