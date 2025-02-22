<a href="https://www.biorxiv.org/content/10.1101/2024.05.28.596164v1"> <img alt="Preprint Badge" src="https://img.shields.io/badge/bioR%CF%87iv-10.1101%2F2024.05.28.596164-red"></a> <a href="https://doi.org/10.5281/zenodo.11402365"><img alt="DOI Badge" src="https://zenodo.org/badge/DOI/10.5281/zenodo.11402365.svg"></a>

> [!NOTE]
> Now [published](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012198) in _PLOS Computational Biology_ 

# Physiology-informed regularization enables training of universal differential equation systems for biological applications
Code for the paper _"Physiology-informed regularisation enables training of universal differential equation systems for biological applications"_



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
- `competitive-inhibition.jl`: Additional experiment on a more complicated competitive inhibition model

The folder also contains the following subdirectories:
- `saved_runs`: Contains the results of the experiment.
- `saved_runs_2`: Contains the results of experiment 2 (more noise)
- `saved_runs_3`: Contains the results of the experiment with longer sampling durations (200 and 400 minutes)
- `saved_runs_ci`: Contains the results of the competitive inhibition experiment.

### Minimal Model
The Minimal Model experiment is implemented in the `minimal-model` directory. The code is divided into the following files:
- `inputs.jl`: Code for generating the input data for the experiment.
- `ude.jl`: Utility functions for the Universal Differential Equation (UDE) model.
- `main.jl`: Code for running the experiment.

The folder also contains the following subdirectories:
- `saved_runs`: Contains the results of the experiment.
- `data`: Contains the data for the experiment.

#### Data Source
The glucose and insulin data files are originally from:

Berry, S.E., Valdes, A.M., Drew, D.A. et al. Human postprandial responses to food and potential for precision nutrition. _Nat Med_ **26**, 964–973 (2020). [https://doi.org/10.1038/s41591-020-0934-0](https://doi.org/10.1038/s41591-020-0934-0)

### Postprocessing
The postprocessing code is implemented in the `post` directory. The code is divided into the following files:
- `generate_figures.jl`: Code for generating the figures for the experiments.


## Running the Code
To activate the Julia environment, make sure you have Julia version `1.10.0` or `1.10.4` installed and run the following commands

### Installing the dependencies
The code dependencies including their versions are included in the `Project.toml` file. When instantiating the environment, the Julia package manager automatically takes care of the appropriate versions. 

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

## Citation
If you use this code in your work, please cite:

de Rooij M, Erdős B, van Riel NAW, O’Donovan SD (2025) Physiology-informed regularisation enables training of universal differential equation systems for biological applications. PLoS Comput Biol 21(1): e1012198. https://doi.org/10.1371/journal.pcbi.1012198

BibTex:
```
@article{rooij_physiology-informed_2025,
	title = {Physiology-informed regularisation enables training of universal differential equation systems for biological applications},
	volume = {21},
	issn = {1553-7358},
	url = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012198},
	doi = {10.1371/journal.pcbi.1012198},
	abstract = {Systems biology tackles the challenge of understanding the high complexity in the internal regulation of homeostasis in the human body through mathematical modelling. These models can aid in the discovery of disease mechanisms and potential drug targets. However, on one hand the development and validation of knowledge-based mechanistic models is time-consuming and does not scale well with increasing features in medical data. On the other hand, data-driven approaches such as machine learning models require large volumes of data to produce generalisable models. The integration of neural networks and mechanistic models, forming universal differential equation (UDE) models, enables the automated learning of unknown model terms with less data than neural networks alone. Nevertheless, estimating parameters for these hybrid models remains difficult with sparse data and limited sampling durations that are common in biological applications. In this work, we propose the use of physiology-informed regularisation, penalising biologically implausible model behavior to guide the UDE towards more physiologically plausible regions of the solution space. In a simulation study we show that physiology-informed regularisation not only results in a more accurate forecasting of model behaviour, but also supports training with less data. We also applied this technique to learn a representation of the rate of glucose appearance in the glucose minimal model using meal response data measured in healthy people. In that case, the inclusion of regularisation reduces variability between UDE-embedded neural networks that were trained from different initial parameter guesses.},
	language = {en},
	number = {1},
	journal = {PLOS Computational Biology},
	author = {Rooij, Max de and Erdős, Balázs and Riel, Natal A. W. van and O’Donovan, Shauna D.},
	month = jan,
	year = {2025},
	note = {Publisher: Public Library of Science},
	keywords = {Blood plasma, Differential equations, Glucose, Insulin, Integrative physiology, Machine learning, Neural networks, Simulation and modeling},
	pages = {e1012198},
}
```
