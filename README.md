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


