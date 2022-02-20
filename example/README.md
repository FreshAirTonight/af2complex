## Example of using AF2Complex


The main script for the example is the shell script `run_af2comp.sh`.
First, use a text editor to open this script, and modify it to provide the correct
python environment of AlphaFold, if not already loaded. Then, provide the correct path to the parameter files of AlphaFold neural network
models. In AF's releases, these models are named as `model_x_ptm.npz` or `model_x_multimer.npz`. Note that AlphaFold-multimer's models (version 2.1.1) are not required if you only wish to run the original monomer DL models.

```sh
./run_af2comp.sh
```

### Files and directories

- `run_af2comp.sh` The main shell script of the example
- `test.lst`  Input targets of the test run
- `af_fea`  Directory contains pre-generated features of monomers and complex targets
- `af_mod`  Examples of output models
- `af2c_mod` Output directory created by running the `run_af2comp.sh` script.
- `run_relaxation.sh` An example shell script of model relaxation.
- `run_fea_gen.sh` An example shell script of input feature generation.
