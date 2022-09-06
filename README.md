# Replication code for Learning from a Biased Sample


This repository contains the code for replicating the figures and simulations in Learning from a Biased Sample, including an implementation of Rockafellar-Uryasev (RU) Regression.

To setup the environment for running the code, run

```
conda env create -n ru_reg --file environment.yml
conda activate ru_reg
```

To reproduce Figure 1, run paper-figure-1.ipynb.

To reproduce Figure 2 and Table 1, run the following commands

```
chmod +x run_one_dim.sh
./run_one_dim.sh
./run_inference.sh
```

and run paper-figure-2.ipynb and paper-table-1.ipynb.

To reproduce Figure 2, run the following commands.

```
chmod +x run_high_dim.sh
./run_high_dim.sh
```
and run paper-table-2.ipynb.



