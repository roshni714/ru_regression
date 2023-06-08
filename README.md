# Replication code for Learning from a Biased Sample

**UPDATE:** We include a stand-alone [demo notebook](demo.ipynb) that can be run with minimal dependencies (Pytorch, numpy, matplotlib). Thanks to Alberto Arletti for contributing this!  

This repository contains the code for replicating the figures and simulations in Learning from a Biased Sample, including an implementation of Rockafellar-Uryasev (RU) Regression.

To setup the environment for running the code, edit the ``prefix``field of the environment.yml file to point to the directory of your conda environments. Then, run
```
conda env create -n ru_reg --file environment.yml
conda activate ru_reg
mkdir figs
mkdir results
```

To reproduce Figure 1, run paper-figure-1.ipynb.

To reproduce Figure 2 and Table 1, edit `MODEL_PATH` and `RUN_PATH` in `run_one_dim.sh` and `MODEL_PATH` in `run_inference.sh` to the directories where you would like to save the trained models and logs, respectively. Then, run the following commands
```
chmod +x run_one_dim.sh
./run_one_dim.sh
./run_inference.sh
```

and run paper-figure-2.ipynb and paper-table-1.ipynb.

To reproduce Table 2, edit `MODEL_PATH` and `RUN_PATH` in `run_high_dim.sh`. Then, run the following commands.

```
chmod +x run_high_dim.sh
./run_high_dim.sh
```
and run paper-table-2.ipynb.

To cite our work:
```
@article{sahoo2022learning,
  title={Learning from a Biased Sample},
  author={Sahoo, Roshni and Lei, Lihua and Wager, Stefan},
  journal={arXiv preprint arXiv:2209.01754},
  year={2022}
}
```


