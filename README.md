> # Project Description
> This repository contains the implementation of the Bipartite Randomized Response mechanism and its application with other DP mechanisms in different scenarios. It is designed to support research and experimental reproducibility in privacy-preserving data analysis.

> # Usage
> The `code` folder contains four subfolders: BRR, DPBoost, L-SRR, and SGDandDNN.
>-   In the `BRR` folder, `LocalRS.py` implements Algorithm 1, which searches for a locally high-utility value of mm starting from a local perspective, and `BRR.py` implements the Bipartite Randomized Response mechanism.
>-   The `DPBoost` folder contains code that applies DP mechanisms to decision trees. Experimental datasets are located in `DPBoost\data\regression`.
> -   The `L-SRR` folder contains a reproduction of the L-SRR scheme proposed by Wang et al. `encode.py` is the encoding script, and running `partition.py` will generate the spatial partitioning results.
> -   The `SGDandDNN` folder includes implementations of various DP mechanisms applied to Stochastic Gradient Descent (SGD) and Deep Neural Networks (DNN).
>#  Illustrate
>Each folder can run independently. For input/output formats and parameter settings, see the comments in the individual scripts. Python 3.12 is recommended and make sure to install the necessary dependency libraries (such as NumPy, Scikit-learn, PyTorch, or TensorFlow, etc.).
