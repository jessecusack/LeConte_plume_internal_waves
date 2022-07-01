#!/usr/bin/env bash
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate lciw && jupyter kernelspec uninstall lciw && conda deactivate
conda remove --name lciw --all