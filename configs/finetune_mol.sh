#!/bin/bash

# Set the path to the YAML configuration file
config_path='/home/shaoqi/code/HAGCL_test/configs/pretrain_mol.yaml'

# List of datasets
datasets=("tox21" "toxcast" "sider" "clintox" "muv" "hiv" "bace" "bbbp")
# datasets=("hiv" "bace" "bbbp")

# Range of seeds
seed_range=(1 2 3 4 5)

# Iterate over datasets
for dataset in "${datasets[@]}"; do
  # Iterate over seeds
  for seed in "${seed_range[@]}"; do
    # Modify the YAML file for the current dataset and seed
    sed -i "/^finetune:/,/^[^ ]/{/runseed:/s/[0-9]\+/$seed/}" "$config_path"
    sed -i "/^finetune:/,/^[^ ]/{/dataset:/s/.*/  dataset: $dataset/}" "$config_path"

    # Execute the Python script with the modified YAML file
    python3 /home/shaoqi/code/HAGCL_test/finetune_mol.py "$config_path"
  done
done
