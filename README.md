This is the experiment environment to measure the performance of the random walk
engine ScaleRunner including all experiments run for our competitor KnightKing.
We define all our experiments using
[Simexpal](https://github.com/hu-macsy/simexpal) (see
[`experiments.yml`](experiments.yml)).

The following steps to reproduce our experiments have been tested on Linux. For
other platforms the steps are likely the same, however, dependencies to OpenMP
or MPI may have to be resolved differently which we will not cover here.

# Installation

## SimexPal

Please install the latest version of simexpal:

```bash
pip install git+https://github.com/hu-macsy/simexpal.git
```

## Configure, Compile and Link all Targets

This step may also resolve any unresolved external dependencies such as cloning
   repositories from GitHub:

```shell
simex develop
```

## Install all Instances

Install all instances using the given URLs in the
[`experiments.yml`](experiments.yml). You can check if you've installed all
instances using:

```shell
simex instances list
```

In order to install the required graph instances you must download the original
files defined under key `instances` in [`experiments.yml`](experiments.yml).
We've put the URLs to download the graph files next to each instance. The graph
file (symbolic link) must either be located in (/instances/). **Important**: an
instance file must be named exactly as it is named in under key `name`.

For example, we added the instance [web-NotreDame](/instances/web-NotreDame)
both in the original form as well as in the converted form.

### Convert Files

Run the [`convert.py`](convert.py) Python script to convert the graphs to our
GDSB binary format:


```shell
python3 convert.py -o ./instances/
```

The script reads the [`experiments.yml`](experiments.yml) to get the graph
properties from the instance `extra_args` property.
- If a graph is directed, add the `--directed` flag. Per default, the graphs are
  considered undirected.
- If a graph is weighted, add the `--weighted` flag. Per default, the graphs are
  considered unweighted.

Using [`GDSB`](https://github.com/hu-macsy/graph-ds-benchmark), the python
script  converts the graph data, writes the binary data to a given output
directory (the provided output path can be absolute or relative) and then
creates symbolic links to that binary data in the instance directory.

# Experiments

## Local Execution

To run the experiments on a machine without job scheduling software one can
simply launch all experiments using:

```shell
simex experiments launch
```

## On Compute Clusters

For compute clusters where the software stack can be loaded using modules, and
job scheduling software the procedure changes a bit. For example, you may need
to load the following modules in order to configure, compile, link and run the
experiments:

```shell
module load tools/gcc/12.3
module load mpi/mpich/4.2.0
```

To schedule experiments using slurm, and in this case a partition named _core_:

```shell
simex e launch --launch-through=slurm --queue=core
```

# Graphanalysis

The [`graphanalysis`](develop/graphanalysis/) tool prints out various metrics
and graph parameters about a graph. The output includes:

- graph name and file
- graph parameters (is the graph weighted, directed, or timestamped)
- count of vertices and edges read
- min. and max. vertex degree (and the associated vertex ID)
- vertex degree distribution.

For the vertex degree distribution the vertices are split into `--data-points`
many chunks. For each chunk a representative vertex is chosen. The chosen
representative vertex ID and its degree are then added to the
`vertex_id_degree_distribution` and `degree_degree_distribution` lists.

To run graphanalysis on a specific graph, first compile and build the tool using

```shell
simex develop graphanalysis
```

and then run it using

```shell
./dev-builds/graphanalysis/bin/graphanalysis -p ./instances/web-NotreDame.data --data-points 200
```

You can also run the tool on all our specified graph instance sets using
simexpal:

```shell
simex e launch --experiment graphanalysis
```

# Branches

- `main`: Representing the latest and stable code base of this project.