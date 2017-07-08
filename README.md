# Unsupervised Handwritten Digit Classification using Spiking Neural Networks and Spike Timing-Dependent Plasticity

This repository contains the code for the plasticity and learning project in the BINDS laboratory at UMass Amherst.

Our code is built on Peter Diehl's [research code](https://github.com/peter-u-diehl/stdp-mnist) and his work with Matthew Cook at ETH Zurich, from the paper [Unsupervised learning of digit recognition using spike-timing-dependent plasticity](http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/full#).

## Setting Things Up

First things first, download the MNIST handwritten digt dataset by running the bash script entitled `get_MNIST.sh` found in the `data` folder. You can run this file by entering `./get_MNIST.sh` in the terminal; if you don't have permission to, execute `chmod +x get_MNIST.sh`.

__Important__: Make sure to change your matplotlibrc file (typically located in `~/.config/matplotlib/`) to include the line `backend : TKAgg`; otherwise, you may not be able to import `pylab` while importing `brian 1.4.3`. The default backend, `GKAgg`, will not work with this outdated module!

Clone this repository to a directory of your choice, denoted by `<stdp-mnist>`. We recommend that your install [Anaconda2](https://www.continuum.io/downloads) and make it your default Python, via adding the line `export PATH=[path to anaconda]/bin:$PATH` to your `.bashrc` (for \*nix systems), as this will make installing the appropriate packages easier. Change directory into `<stdp-mnist>`, and execute `pip install -r requirements.txt` in a terminal (on \*nix systems) or a Git Bash shell (on Windows), and then execute `conda install -r conda_requirements.txt`.

To train a network, change directory into `<stdp-mnist>/code/train`, and choose one of the scripts therein to train a spiking neural network model on the MNIST handwritten digit dataset. We recommend you train a convolutional spiking neural network with between-patch connectivity (CSNN-PC), which is the most general model in that it can be reduced to a convolutional spiking neural network without between-patch connectivity (CSNN) or simply the spiking neural network model (SNN) from the above [paper](http://journal.frontiersin.org/article/10.3389/fncom.2015.00099/full#) in certain special cases. To do so, simply run

```
python csnn_pc_mnist.py
```

to train a CSNN-PC model with default parameters. To see a list of optional arguments to specify on the command line, run

```
python csnn_pc_mnist.py --help
```

Perhaps the most illuminating command-line argument is `do_plot`, which, when set as `do_plot=True`, allows you to visualize network training progress, learned convolution filters and between-patch connection weights, current input to the network, and the distribution of "votes" (i.e., individual excitatory neuron classifications) over the last minibatch (`update_interval`) of data.

There are various supporting scripts in other subfolders of the project repository which allow a user accomplish tasks like testing or visualizing parameters of trained models, plotting performance curves, and running jobs on the CICS swarm2 computing cluster. As the project is currently under heavy development, much of the code is fragmented, disorganized, and / or broken, but the state of things will hopefully improve as the project develops.
