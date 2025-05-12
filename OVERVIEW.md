In this document we will provide all necessary steps to reproduce our
experiments and evaluate the results specifically for the artifact provided to
the conference EuroPar'25. Compared to the [Readme File](README.md), this
document will give a much more thorough explanation on what is needed to
configure, compile, link and run the experiments. 

Please consider that we conducted our experiments on a cluster with 16 compute
nodes, each equipped with 2x 12-Core Intel Xeon X6126 (HT) CPUs, and 192 GB RAM
inter-connected by a 100 GBit Infiniband Omnipath network. C++ code was compiled
using GCC v12.3, MPICH v4.2.0, and Intel Threading Building Blocks (TBB)
v2021.11.

Running all experiments with the original graph files can take more than 24
hours. Therefore, we provide a small [example graph data set](/instances/) which
we will use to showcase all conducted experiments, as well as plotting the
measurements.

The whole experiment setup uses [Simexpal](https://github.com/hu-macsy/simexpal)
which is [well documented](https://simexpal.readthedocs.io/en/latest/) in case
you experience any issues we have not covered.

# Getting Started

The following software was used to conduct and evaluate our experiments:

- SimexPal (commit c848baba0baa8e9794bdcc6b9c5d2a507a840953)
- C++ Compiler GCC (v12.3) with (minimum) support of C++17
- MPICH (v4.2.0)
- Python3 (v3.12.3)
- [Python Jupyter Notebook](https://jupyter.org/install)

To install python packages you may want to activate your python virtual
environment.

To install the right version of SimexPal you can use pip:

```bash
pip install git+https://github.com/hu-macsy/simexpal.git@c848baba0baa8e9794bdcc6b9c5d2a507a840953
```

Or you install SimexPal by cloning the repository:

```bash
git clone https://github.com/hu-macsy/simexpal.git
cd simexpal
git checkout c848baba0baa8e9794bdcc6b9c5d2a507a840953
pip install -e .
```

Consider reading the [quick start guide for
SimexPal](https://simexpal.readthedocs.io/en/latest/quick_start.html) if you
have issues with the installation or the usage of the tool.

Please unpack the provided compressed artifact and navigate into its root
directory.

First, we will need to configure, compile and link all builds defined in our
[experiments.yml file](experiments.yml):

```bash
simex develop
```

You may encounter two issues here:

1. Your C++ compiler does not support certain functionalities needed by our
   programs. To fix this, you can export the required C++ compiler. For example,
   for the GNU v12 C++ compiler *G++12* use: `export CXX=/usr/bin/g++-12`. In
   case you have already tried to build using `simex develop`: reconfigure using
   `simex develop --reconfigure` instead.

2. The `scalerunnerConfig.cmake` file cannot be found since you did not provide
   the  `scalerunner_DIR` to the build `io_benchmark`. Please navigate into the
   `experiments.yml` file under the build `io_benchmark`. There you will find
   the configure steps for the build. The `scalerunner_DIR` is either provided
   using `/lib/...` or `/lib64/...`. Please comment in/out one of the lines
   depending on the requirements of your system. This may also be the case for
   build `sr_benchmark`. Reconfigure using `simex develop --reconfigure`
   instead.

Next, we need to convert the [instance files](/instances/) using the [python
convert script](convert.py). Either call the convert script as described in the
[experiments.yml file](experiments.yml) under key `instances` or you use
SimexPal:

```bash
simex instances install
```

Finally, we can run all experiments! In this case we are using the option
`--launch-through=fork` to run one experiment after the other in the same
process as the one calling simex: 

```bash
simex e launch --launch-through=fork
```

MPI clusters often require different runtimes to execute MPI programs. If you
need (or want) to change the runtime that executes the programs using MPI,
namely the programs `kk_benchmark`, `sr_benchmark`, and `io_benchmark` you will
have to adapt the [experiments.yml file](experiments.yml) under the key
`experiments`, where you find the arguments passed to call the program. There
are also instructions to adapt the arguments to your systems requirements. 

After launching all experiments, you can list all experiments and their status
using:

```bash
simex e list
```

Which should now look similar to:

```bash
Experiment                     started    finished   failures             other
----------                     -------    --------   --------             -----
graphanalysis @ _dev                      4/4                             
graphfile-input ~ gdsb-mpi-io             4/4                             
graphfile-input ~ original @ _            4/4                             
kk-crw ~ 80 @ _dev                        4/4                             
kk-node2vec ~ 80, homopholy @             4/4                             
kk-node2vec ~ 80, structure @             4/4                             
scalerunner-crw ~ 80 @ _dev               4/4                             
scalerunner-node2vec ~ 80, hom            4/4                             
scalerunner-node2vec ~ 80, str            4/4                             
scalerunner-scaling1 ~ 80 @ _d            4/4                             
scalerunner-scaling16 ~ 80 @ _            4/4                             
scalerunner-scaling2 ~ 80 @ _d            4/4                             
scalerunner-scaling4 ~ 80 @ _d            4/4                             
scalerunner-scaling8 ~ 80 @ _d            4/4                             
56 experiments in total
```

The output tells us that for all defined experiments we have run 4 experiments
and all 4 finished (successfully).

If there have been any failed runs and you would like to re-run them you can
purge all experiments:

```bash
simex e purge --all -f
```

All output files have been written to the [output folder](/output/).

If you ran the experiments on another machine and you would like to archive and
copy the results to another, first archive the data:

```bash
simex archive
```

This will compress the `output folder` and the `experiments.yml` file into
`data.tar.gz`. Copy this file to the machine you want to evaluate the results
on, and extract the contents to the top level such that the directory `output`
is on the same level as the `experiments.yml` and `evaluation.ipynb` file.

We can now evaluate our experiments using the [evaluation jupyter notebook
script](evaluation.ipynb). Remember: this requires you to have installed a
[Jupyter Notebook Environment](https://jupyter.org/install).

First, the notebook defines all imports, and therefore all python 3 packages you
will need to install. Simply run all cells of the notebook, you will not need to
change anything of the script. If you do not want to write all plots to a PDF
file, change the global variable `save_all_plots=True` to
`save_all_plots=False`.

Please note that the plotted results are not to be interpreted in any way, our
example data set is just enough to provide a running example with very small
graph instances.

# Step-by-Step Instructions

In order to reproduce our experiments, the workflow is *nearly* the same as
described in the getting started. 

First, download all instances defined under the key `instances` in the
[experiments.yml file](experiments.yml) either directly to the [instance
files](/instances/) or any other folder, but do not forget to create symbolic
links to the files (without the file ending such as `.edges` or `.mtx`). 

You can also choose only certain instance sets which then have to be specified
under the key `matrix` in the [experiments.yml file](experiments.yml), where
each experiment is provided a list of instance sets under the key `instsets`.
You will find the original `instsets` list there as well which we will now need
to comment in, and comment out the `example_graphs` instance set.

Once you downloaded and converted all files defined in `instsets`, the list of
instances should mark all instances in green when using:

```bash
simex e list
```

To run all experiments using slurm using the *queue* (also known as partition)
core:

```bash
simex e launch --launch-through=slurm --queue=core
```

Once all experiments are finished, you can *archive* and copy the results as
described above. Finally you can evaluate the results using the [evaluation
jupyter notebook script](evaluation.ipynb).

We provide the [original experimental results data
set](/archives/experimental_resutls-EUROPAR25.zip) which was used to plot the
data of our paper. In order to plot the results: unzip the file into the root
directory of this project. Doing so will write an `output` directory and an
`experiments.yml` file. The `experiments.yml` file represents the original
`experiments.yml` file used to run and evaluate our experiments. For examples
and further documentation we have made small changes to the `experiments.yml`
file provided in this artifact, so simply rename the current `experiments.yml`
file to `experiments.yml.tmp`. The following commands will do all these steps:

```bash
rm -rf output
mv experiments.yml experiments.yml.tmp
unzip archives/experimental_resutls-EUROPAR25.zip
mv experimental_resutls-EUROPAR25/* .
```

Now you can run the evaluation script which will plot the same figures as we
have in our paper.