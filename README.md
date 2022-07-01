# Leconte Plume Internal Waves

Analysis of plume generated internal gravity waves at LeConte, Alaska.

## Requirements

* UNIX-like operating system such as macOS or some flavour of LINUX
* git and wget installed on your system
* The [conda package manager](https://conda.io/en/latest/) (I recommend the lightweight version [miniconda](https://docs.conda.io/en/latest/miniconda.html))
* MATLAB version R2020a or greater

## Installing and removing the environment

A conda environment is specified in `environment.yml` and may be install using the appropriate bash scripts. 

To install:

```bash
./install_environment.sh
```

To remove:

```bash
./remove_environment.sh
```

These also install/remove the jupyter kernel for the environment.

> If these don't execute, you might need to change the file permissions with `chmod u+x *.sh`.

## Project Structure
```
LeConte_plume_internal_waves/
    ├── LICENSE
    ├── README.md          <- The top-level README for people using this project.
    ├── AUTHORS.md         <- Author information.
    ├── data/
    │   └── README.md      <- Information on data sources and retrieval. 
    │
    ├── analysis/          <- Jupyter notebooks, MATLAB code and anything else that constitutes analysis.
    │   ├── README.md      <- Any information about the analysis, such as execution order. 
    │   ├── *.py           <- Python files that can be converted to notebooks using jupytext.
    │   └── *.m            <- Analysis in MATLAB.
    │
    ├── figures/           <- Saved figures generated during analysis.
    │
    ├── environment.yml    <- Conda environment specification. Install using the bash scripts.
    │
    ├── matlab_toolboxes/  <- A place for 3rd party MATLAB toolboxes.
    │   ├── toolbox/
    │   │
    │   └── get_toolbox.sh <- Script to download toolboxes.
 ```

---

* Free software: MIT license

