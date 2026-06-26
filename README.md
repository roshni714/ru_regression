# Replication code for Learning from a Biased Sample

**UPDATE:** We include a stand-alone [demo notebook](demo.ipynb) that can be run on CPU with minimal dependencies (Pytorch, numpy, matplotlib). Thanks to Alberto Arletti for contributing this!  

This repository contains the code for replicating the figures and simulations in Learning from a Biased Sample, including an implementation of Rockafellar-Uryasev (RU) Regression.

## Data
The MIMIC-III dataset can be obtained upon request [here](https://physionet.org/content/mimiciii/1.4/). The BRFSS 2021 dataset is publicly available [here](https://www.cdc.gov/brfss/annual_data/annual_2021.html), and the 2018 BRFSS Depression and Anxiety Module used in the paper is publicly available [here](https://www.cdc.gov/brfss/questionnaires/modules/category2018.htm). The HPS 2021 dataset is publicly available [here](https://www.census.gov/programs-surveys/household-pulse-survey/datasets.html). We provide R scripts for prepping the datasets in `data_preprocessing/`. A google drive folder with the raw and cleaned BRFSS and HPS datasets is provided [here](https://drive.google.com/drive/folders/1P8dnEGHoBycaVyp9iVcy5gpDc6wroOIh?usp=drive_link). Data dictionaries are available in in `data_dictionaries/`.

## Environment
To setup the environment for running the code, edit the ``prefix``field of the environment.yml file to point to the directory of your conda environments. Then, run
```
conda env create -n ru_reg --file environment.yml
conda activate ru_reg
mkdir figs
mkdir results
```

## Configuration
Next, update paths to data files in `data_loaders/survey_dataloader.py` and `data_loaders/mimic.py` and `OUTPUT_PATH` and `SAVE_PATH` in `generate_sbatches.py.`

## Reproduction of Paper Figures
To reproduce figures from paper, run the following command to launch all experiments. After experiments are complete, run the associated notebook.
```
chmod +x run_scripts.sh
./run_scripts.sh
```
## Contents
The replication package contains the following files and folders:

```
ru_regression/
│
├── data_loaders/
├── data_preprocessing/
├── modules/
│
├── train.py
├── loss.py
├── reporting.py
├── utils.py
├── generate_sbatches.py
├── run_scripts.sh
│
├── demo.ipynb
├── figure-1-paper.ipynb
├── figure-3-paper.ipynb
├── ...
├── figure-8-paper.ipynb
│
├── environment.yml
└── README.md
```
- Main RU Regression learning methodology implemented in
  - `loss.py` - RU loss implementation
  - `modules/` - Model architectures
  - `train.py` - training pipeline
- Data preparation.
	- `data_preprocessing/`: Contains R scripts for cleaning raw data files.
	- `data_loaders/`: Contains dataset-specific loading utilities. Supports datasets including: MIMIC-III, BRFSS 2021, BRFSS Depression and Anxiety Module, HPS 2021
- Replication of paper figures. The repository includes Jupyter notebooks used to reproduce results from the paper:
  - `figure-1-paper.ipynb`
  - `figure-3-paper.ipynb`
  - `figure-4-paper.ipynb`
  - `figure-5a-paper.ipynb`
  - `figure-5b-paper.ipynb`
  - `figure-6-paper.ipynb`
  - `figure-7-paper.ipynb`
  - `figure-8-paper.ipynb`
- Minimal example demonstrating RU regression: `demo.ipynb` 

## Computation
Each script can be run with 1 CPU and 1 GPU with 20 GB of RAM for a maximum of 1 hour total. Each script completes in roughly 15 minutes.

To cite our work:
```
@article{sahoo2022learning,
  title={Learning from a Biased Sample},
  author={Sahoo, Roshni and Lei, Lihua and Wager, Stefan},
  journal={arXiv preprint arXiv:2209.01754},
  year={2022}
}
```


