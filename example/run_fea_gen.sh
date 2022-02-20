#!/bin/bash
# An example script of feature generation. This heavily depenedent on your installation,
# due to many third-party tools and multiple sequence libraries.
#
# You need to take care of these paths, python environment, and third-party sequence tools.

#. load_alphafold  ## set up proper AlphaFold conda environment.

DATA_DIR=$HOME/scratch/afold/data
export HHLIB=$HOME/data/tools/hh-suite/build
export HMMER=$HOME/data/tools/hmmer-3.2.1/build
export KALIGN=$HOME/data/tools/kalign_v2/kalign
af_dir=../src

if [ $# -eq 0 ]
  then
    echo "Usage: $0 <seq_file>"
    exit 1
fi
fasta_path=$1
out_dir=af2c_fea
db_preset='reduced_dbs'
model_preset='monomer_ptm'
max_template_date=2022-02-02
is_prokaryote='false'  # only if you run multimer data pipeline

echo "Info: sequence file $seq_file"
echo "Info: target_lst $target_lst"
echo "Info: out_dir $out_dir"
echo "Info: alphafold db preset $db_preset"
echo "Info: alphafold model preset $model_preset"
echo "Info: max_template_date is $max_template_date"


##########################################################################################


if [ "$model_preset" = "multimer" ]; then
  python $af_dir/run_af2c_fea.py --fasta_paths=$fasta_path --db_preset=$db_preset \
    --data_dir=$DATA_DIR --output_dir=$out_dir      \
    --uniprot_database_path=$DATA_DIR/uniprot/uniprot.fasta \
    --uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
    --mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters.fa   \
    --pdb_seqres_database_path=$DATA_DIR/pdb_seqres/pdb_seqres.txt \
    --small_bfd_database_path=$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta \
    --template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files  \
    --max_template_date=$max_template_date                 \
    --obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
    --hhblits_binary_path=$HHLIB/bin/hhblits   \
    --hhsearch_binary_path=$HHLIB/bin/hhsearch \
    --jackhmmer_binary_path=$HMMER/bin/jackhmmer \
    --hmmsearch_binary_path=$HMMER/bin/hmmsearch \
    --hmmbuild_binary_path=$HMMER/bin/hmmbuild \
    --kalign_binary_path=$KALIGN \
    --is_prokaryote_list=$is_prokaryote \
    --model_preset=$model_preset
else
  python $af_dir/run_af2c_fea.py --fasta_paths=$fasta_path --db_preset=$db_preset \
    --data_dir=$DATA_DIR --output_dir=$out_dir      \
    --uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
    --mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters.fa   \
    --small_bfd_database_path=$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta \
    --pdb70_database_path=$DATA_DIR/pdb70/pdb70           \
    --template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files  \
    --max_template_date=$max_template_date                \
    --obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
    --hhblits_binary_path=$HHLIB/bin/hhblits   \
    --hhsearch_binary_path=$HHLIB/bin/hhsearch \
    --jackhmmer_binary_path=$HMMER/bin/jackhmmer \
    --hmmsearch_binary_path=$HMMER/bin/hmmsearch \
    --hmmbuild_binary_path=$HMMER/bin/hmmbuild \
    --kalign_binary_path=$KALIGN \
    --model_preset=$model_preset
  fi
