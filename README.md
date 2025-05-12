This is the experiment environment to measure the performance of the random walk
engine ScaleRunner including all experiments run for our competitor KnightKing.
We define all our experiments using
[Simexpal](https://github.com/hu-macsy/simexpal) (see
[`experiments.yml`](experiments.yml)).

We provide an overview document on how to run experiments in the [overview
document](/overview/scalerunner_exp_overview.pdf).

# On Compute Clusters

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
- `artifact`: A special artifact branch for the EuroPar'25 artifact submission.