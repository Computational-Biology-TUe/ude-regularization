<a href="https://www.biorxiv.org"> <img alt="bioRXiv Badge" src="https://img.shields.io/badge/bioR%CF%87iv-pending-red?style=flat"></a> <a href="https://doi.org/10.5281/zenodo.11402366"><img alt="DOI Badge" src="https://zenodo.org/badge/DOI/10.5281/zenodo.11402366.svg"></a>


# Physiology-informed regularization enables training of universal differential equation systems for biological applications
Code for the paper _"Physiology-informed regularization enables training of universal differential equation systems for biological applications"_

**Authors**: <u>[Max de Rooij](https://orcid.org/0009-0006-1298-7385),</u> [Balázs Erdős](https://orcid.org/0000-0001-8643-4915), [Natal van Riel](https://orcid.org/0000-0001-9375-4730) & [Shauna O'Donovan](https://orcid.org/0000-0003-2253-4903)


## Code Structure
The code is divided into the following directories:
1. `michaelis-menten`: Contains the code for the Michaelis-Menten experiment.
2. `minimal-model`: Contains the code and data for the Minimal Model experiment.
3. `post`: Contains the code for postprocessing the results of the experiments and generating the figures.

### Michaelis-Menten
The Michaelis-Menten experiment is implemented in the `michaelis-menten` directory. The code is divided into the following files:
- `inputs.jl`: Code for generating the input data for the experiment.
- `ude.jl`: Utility functions for the Universal Differential Equation (UDE) model.
- `main.jl`: Code for running the experiment.

The folder also contains the following subdirectories:
- `saved_runs`: Contains the results of the experiment.

### Minimal Model
The Minimal Model experiment is implemented in the `minimal-model` directory. The code is divided into the following files:
- `inputs.jl`: Code for generating the input data for the experiment.
- `ude.jl`: Utility functions for the Universal Differential Equation (UDE) model.
- `main.jl`: Code for running the experiment.

The folder also contains the following subdirectories:
- `saved_runs`: Contains the results of the experiment.
- `data`: Contains the data for the experiment.

### Postprocessing
The postprocessing code is implemented in the `post` directory. The code is divided into the following files:
- `generate_figures.jl`: Code for generating the figures for the experiments.


## Running the Code
To activate the Julia environment, make sure you have Julia version 1.10 installed and run the following commands

### Installing the dependencies
1. Open the Julia REPL
```bash
julia
```

2. Press `]` to enter the package manager mode and activate the environment
```julia
activate .
```

3. Instantiate the environment
```julia
instantiate
```

### Running the experiments
To run the experiments, run the `main.jl` file in the respective experiment directory.

### Generating the figures
To generate the figures, run the `generate_figures.jl` file in the `post` directory. Figures are then saved in the `figures` directory.
