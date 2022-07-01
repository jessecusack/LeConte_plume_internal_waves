#!/usr/bin/env bash
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate lciw && python -m ipykernel install --user --name lciw
cd matlab_toolboxes && ./get_toolboxes.sh
