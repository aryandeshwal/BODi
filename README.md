# Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings
- This repository contains source code for the [AISTATS 2023](https://aistats.org/aistats2023/index.html) paper [Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings](https://arxiv.org/abs/2303.01774). 

### How to run the code?
- packages/libraries required to run the code are listed in requirements.txt.
- runMaxSAT.ipynb notebook shows a way to run the code for MaxSAT synthetic function (60 binary variables)
- `bodi/run_experiment.py` is the main entry point for the code
- BODi algorithm is run by calling `run_experiment` method from `bodi/run_experiment.py`
- main parameters required for `run_experiment` are:
    - `n_prototype_vectors`: no of dictionary elements/anchor points used in the surrogate model, we recommend keeping it to 64 or 128.
    - `evalfn`: name of the objective-function/problem to optimize. For a candidate definition, please see LABS/Ackley53/MaxSAT60/SVM in `bodi/test_functions.py`.
    - `max_evals`: maximum number of evaluations/calls to the objective function.
    - `n_initial_points`: no. of input points to initialize the BO surrogate model, we recommended keeping it to 20 unless the evaluation budget is more expensive.
    - `n_binary`: no of binary variables in the input for the given objective, for e.g. MaxSAT has 60 binary variables.
    - `n_continuous`: no of continuous/ variables in the input for the given objective, for e.g. Ackley53 has 3 continuous variables.
    - `n_replications`: number of times to rerun the code, this is mostly useful for generating multiple runs and getting mean and error bars of the BO performance.
    - `batch_size`: no of batch evaluations in each BO round.


- simple way to add a new objective-function/problem is by creating a class similar to LABS or Ackley53 in `bodi/test_functions.py` and then call it in the `run_experiment` method of `run_experiment.py` (line 98-113).
- acquisition function optimization routines for both binary/mixed spaces are defined in  `bodi/optimize.py`.
- dictionary kernel with both diverse random and binary wavelet dictionaries are defined in `bodi/dictionary_kernel.py`.