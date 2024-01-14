#!/bin/bash

# Set the path to the YAML configuration file
config_path='/home/shaoqi/code/HAGCL/configs/train_graph.yaml'

# List of datasets
datasets=("NCI1" "PROTEINS" "DD" "MUTAG" "COLLAB" "REDDIT-BINARY" "REDDIT-MULTI-5K" "IMDB-BINARY")


# Range of seeds
seed_range=(6 7 8 9 0)

# Iterate over datasets
for seed in "${seed_range[@]}"; do
  # Iterate over seeds
  for dataset in "${datasets[@]}"; do
    # Modify the YAML file for the current dataset and seed
    sed -i "/^runseed:/s/[0-9]\+/$seed/" "$config_path"
    sed -i "/^dataset:/s/.*/dataset: $dataset/" "$config_path"

    # Execute the Python script with the modified YAML file
    python3 /home/shaoqi/code/HAGCL/train_graph.py "$config_path"
  done
done


