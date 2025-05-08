#pragma once

#include "graph_io.h"

#include <kklib/graph_data_structure.hpp>
#include <kklib/node2vec.hpp>
#include <kklib/storage.hpp>
#include <kklib/type.hpp>
#include <kklib/walk.hpp>

#include <gdsb/experiment.h>

#include <mpi.h>

#include <chrono>

namespace experiments
{

template <typename Edges, typename EdgeData>
std::chrono::milliseconds
vanilla_node2vec(Edges&& edges, gdsb::Vertex32 const vertex_count, Node2vecConf& node2vec_conf, WalkConfig& walk_conf)
{
    kklib::LoadedGraphData<EdgeData> graph_data;
    graph_data.read_e_num = edges.size();
    graph_data.read_edges = new Edge<EdgeData>[graph_data.read_e_num];
    graph_data.v_num_param = vertex_count;

    benchmark::copy_edges<Edge<EdgeData>, EdgeData>(edges, graph_data.read_edges, graph_data.read_e_num);
    edges.clear();

    WalkEngine<EdgeData, Node2vecState> graph{ graph_data.v_num_param, graph_data.read_edges, graph_data.read_e_num };
    // TODO: It would be better to not use omp here but a function parameter. But for now this will suffice.
    graph.set_concurrency(omp_get_max_threads());

    MPI_Barrier(MPI_COMM_WORLD);
    return gdsb::benchmark(
        [&]()
        {
            node2vec(&graph, node2vec_conf, walk_conf);
            MPI_Barrier(MPI_COMM_WORLD);
            return true;
        });
}

template <typename Edges, typename EdgeData>
std::chrono::milliseconds
crw_vanilla(Edges&& edges, gdsb::Vertex32 const vertex_count, WalkConfig& walk_conf, uint64_t random_walk_length, uint64_t random_walk_count)
{
    kklib::LoadedGraphData<EdgeData> graph_data;
    graph_data.read_e_num = edges.size();
    graph_data.read_edges = new Edge<EdgeData>[graph_data.read_e_num];
    graph_data.v_num_param = vertex_count;

    benchmark::copy_edges<Edge<EdgeData>, EdgeData>(edges, graph_data.read_edges, graph_data.read_e_num);
    edges.clear();

    WalkEngine<EdgeData, EmptyData> graph{ graph_data.v_num_param, graph_data.read_edges, graph_data.read_e_num };
    graph.set_concurrency(omp_get_max_threads());

    WalkerConfig<EdgeData, EmptyData> walker_conf(random_walk_count);

    auto extension_comp = [&](Walker<EmptyData>& walker, VertexID) -> real_t
    { return walker.step >= random_walk_length ? 0.f : 1.f; };

    auto static_comp = get_trivial_static_comp(&graph);
    TransitionConfig<EdgeData, EmptyData> tr_conf(extension_comp, static_comp);

    MPI_Barrier(MPI_COMM_WORLD);
    return gdsb::benchmark(
        [&]()
        {
            graph.random_walk(&walker_conf, &tr_conf, walk_conf);
            MPI_Barrier(MPI_COMM_WORLD);
            return true;
        });
}

} // namespace experiments