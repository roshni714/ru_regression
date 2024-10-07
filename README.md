# Replication code for Learning from a Biased Sample

**UPDATE:** We include a stand-alone [demo notebook](demo.ipynb) that can be run on CPU with minimal dependencies (Pytorch, numpy, matplotlib). Thanks to Alberto Arletti for contributing this!  

This repository contains the code for replicating the figures and simulations in Learning from a Biased Sample, including an implementation of Rockafellar-Uryasev (RU) Regression.

To setup the environment for running the code, edit the ``prefix``field of the environment.yml file to point to the directory of your conda environments. Then, run
```
conda env create -n ru_reg --file environment.yml
conda activate ru_reg
mkdir figs
mkdir results
```

Data: The MIMIC-III dataset can be obtained upon request [here](https://physionet.org/content/mimiciii/1.4/). The BRFSS 2021 dataset is publicly available [here](https://www.cdc.gov/brfss/annual_data/annual_2021.html), and the 2018 BRFSS Depression and Anxiety Module used in the paper is puclicly available [here](https://www.cdc.gov/brfss/questionnaires/modules/category2018.htm). The HPS 2021 dataset is publicly available [here](https://www.census.gov/programs-surveys/household-pulse-survey/datasets.html).

Next, update paths to data files in `data_loaders/survey_dataloader.py` and `data_loaders/mimic.py`.


To reproduce figures from paper, run the following command to launch all experiments. After experiments are complete, run the associated notebook.
```
chmod +x run_scripts.sh
./run_scripts.sh
```

To cite our work:
```
@article{sahoo2022learning,
  title={Learning from a Biased Sample},
  author={Sahoo, Roshni and Lei, Lihua and Wager, Stefan},
  journal={arXiv preprint arXiv:2209.01754},
  year={2022}
}
```


