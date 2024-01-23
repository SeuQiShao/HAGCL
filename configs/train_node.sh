#!/bin/bash

# Set the path to the YAML configuration file
config_path='/home/shaoqi/code/HAGCL_test/configs/train_node.yaml'

# List of datasets
datasets=("Cora" "Citeseer" "Pubmed" "wikics" "amc" "amp" "coc")


# Range of seeds
seed_range=(0 1 2 3 4 5 6 7 8 9)

# Iterate over datasets
for seed in "${seed_range[@]}"; do
  # Iterate over seeds
  for dataset in "${datasets[@]}"; do
    # Modify the YAML file for the current dataset and seed
    sed -i "/^runseed:/s/[0-9]\+/$seed/" "$config_path"
    sed -i "/^dataset:/s/.*/dataset: $dataset/" "$config_path"

    # Execute the Python script with the modified YAML file
    python3 /home/shaoqi/code/HAGCL_test/train_node.py "$config_path"
  done
done


