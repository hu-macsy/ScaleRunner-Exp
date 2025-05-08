#include "experiments.h"
#include "graph_io.h"
#include "helpers.h"

#include <gdsb/experiment.h>

#include <kklib/mpi_helper.hpp>
#include <kklib/node2vec.hpp>
#include <kklib/static_comp.hpp>
#include <kklib/storage.hpp>
#include <kklib/walk.hpp>

#include <CLI/CLI.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

//! Quick launch:
//! Using weighted graph:
//! mpirun -n 2 ./dev-builds/kk_benchmark/bin/kk_benchmark -p instances/aves-barn-swallow-non-physical.data
int main(int argc, char** argv)
{
    kklib::MPI_Instance mpi_instance(&argc, &argv);

    //======================================================================
    // CLI PARSE
    CLI::App app{ "benchmark" };

    // Graph Input Parameters
    std::string graph_file_path_in = "";
    app.add_option("-p,--path", graph_file_path_in, "Full path to graph file.")->required()->check(CLI::ExistingFile);
    uint64_t graph_vertex_count = 0;
    app.add_option("-n, --vertex-count", graph_vertex_count, "Provided count of vertices for given graph.");

    std::string experiment_name = "none";
    app.add_option("-e, --experiment", experiment_name, "Name of the experiment for post processing.");

    bool rw_write_output = false;
    app.add_flag("--write-output", rw_write_output,
                 "Only if this flag is set, the output path will be used and the random walk output will be written.");

    std::string experiment_output_path_input = "./output/walk";
    app.add_option("-o, --output-path", experiment_output_path_input, "Output path for experiment paths.");

    // random walk parameters
    std::string random_walk_algorithm = "node2vec";
    app.add_option("-r, --random-walk-algorithm", random_walk_algorithm, "Name of the random walk algorithm.");

    uint64_t random_walk_length = 50;
    app.add_option("--rw-length", random_walk_length, "Random walk length.");

    // If this is 0, we'll start n random walks implicitly
    uint64_t random_walk_count = 0;
    app.add_option("--rw-count", random_walk_count, "Random walk count.");

    // node2vec parameters
    float node2vec_p = 1.0;
    app.add_option("--node2vec-p,", node2vec_p, "node2vec parameter p.");
    float node2vec_q = 2.0;
    app.add_option("--node2vec-q,", node2vec_q, "node2vec parameter q.");

    // not used in benchmark but in converter:
    uint64_t graph_edge_count = 0;
    app.add_option("-m, --edge-count", graph_edge_count, "Provided count of edges for given graph.");
    std::string graph_file_type = "edges";
    app.add_option("--file-type", graph_file_type, "The graph file format: edges or mtx");
    bool graph_directed = false;
    app.add_flag("-d, --directed", graph_directed, "Flag declaring graph file is directed.");
    bool graph_weighted = false;
    app.add_flag("-w, --weighted", graph_weighted, "Graph file contains weight.");
    bool graph_timestamped = false;
    app.add_flag("-t, --timestamped", graph_timestamped, "Flag declaring graph file carries timestamps.");
    std::string graph_category = "web";
    app.add_option("--graph-category", graph_category, "Specify graph category.");

    //======================================================================
    // Parse Input Parameter
    CLI11_PARSE(app, argc, argv);

    benchmark::RandomWalkAlgorithmParameter const rw_algo_param = benchmark::rw_algorithm(random_walk_algorithm);

    auto error_handler_f = [](std::filesystem::path const& experiment_output_directory)
    {
        if (benchmark::is_master_process())
        {
            std::cerr << "Path: [" << experiment_output_directory
                      << "] is neither a directory, nor could a directory with that name be created." << std::endl;
        }

        std::exit(EXIT_FAILURE);
    };

    std::filesystem::path experiment_output_path =
        benchmark::provide_experiment_output_path(std::move(error_handler_f), rw_write_output,
                                                  experiment_output_path_input, experiment_name);

    std::filesystem::path const file_path(graph_file_path_in);

    if (benchmark::is_master_process())
    {
        // This is due to the implicit use of omp_get_max_threads() in
        // knightking. We don't want to check on scaling for threads, but this
        // should normally be parameterized.
        gdsb::out("thread_count", omp_get_max_threads());

        gdsb::out("mpi_rank", get_mpi_rank());
        gdsb::out("mpi_size", get_mpi_size());

        gdsb::out("graph_file_path_in", graph_file_path_in);
        gdsb::out("graph_name", file_path.filename());
        gdsb::out("graph_category", graph_category);

        gdsb::out("experiment_name", experiment_name);
        gdsb::out("rw_write_output", rw_write_output);
        gdsb::out("experiment_output_path_input", experiment_output_path_input);
        gdsb::out("experiment_output_path", experiment_output_path);
        gdsb::out("random_walk_algorithm", random_walk_algorithm);
        gdsb::out("random_walk_length", random_walk_length);
        gdsb::out("p_return", node2vec_p);
        gdsb::out("q_in_out", node2vec_q);
    }

    //======================================================================
    // Experiments
    std::chrono::milliseconds experiment_duration{ 0 };

    WalkConfig walk_conf;
    if (rw_write_output)
    {
        walk_conf.set_output_file(experiment_output_path.c_str());
    }

    benchmark::MemoryFootprintT mem_before_graph_read_operation = gdsb::memory_usage_in_kb();
    benchmark::MemoryFootprintT mem_before_graph_read_operation_global =
        benchmark::mpi_reduce_memory_footprint(mem_before_graph_read_operation);
    if (benchmark::is_master_process())
    {
        gdsb::out("mem_before_graph_read_operation_global", mem_before_graph_read_operation_global);
    }

    gdsb::WallTimer graph_read_timer;

    graph_read_timer.start();
    benchmark::ReadGraphData read_graph_data = benchmark::read_in_edges_from_file(file_path);
    graph_read_timer.end();

    std::chrono::nanoseconds const graph_read_duration_ns = graph_read_timer.duration();
    std::chrono::milliseconds const graph_read_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(graph_read_duration_ns);

    benchmark::MemoryFootprintT mem_after_graph_read_operation = gdsb::memory_usage_in_kb();
    benchmark::MemoryFootprintT mem_after_graph_read_operation_global =
        benchmark::mpi_reduce_memory_footprint(mem_after_graph_read_operation);
    benchmark::MemoryFootprintT mem_after_graph_read_operation_max =
        benchmark::mpi_max_memory_footprint(mem_after_graph_read_operation);
    if (benchmark::is_master_process())
    {
        gdsb::out("graph_read_duration_ms", graph_read_duration_ms.count());
        gdsb::out("mem_after_graph_read_operation_global", mem_after_graph_read_operation_global);
        gdsb::out("mem_after_graph_read_operation_max", mem_after_graph_read_operation_max);
    }

    auto rw_count_error_f = []()
    {
        if (benchmark::is_master_process())
        {
            std::cerr << "Can not determine random walk count due to missing random walk count and graph vertex count "
                         "parameters."
                      << std::endl;
        }

        std::exit(EXIT_FAILURE);
    };
    random_walk_count = benchmark::check_random_walk_count_parameters(std::move(rw_count_error_f), random_walk_count,
                                                                      read_graph_data.file_header.vertex_count);

    // Now we print the graph meta data, and the meta data associated to this
    // MPI process aka partition.
    if (benchmark::is_master_process())
    {
        gdsb::out("vertex_count_in", read_graph_data.file_header.vertex_count);
        gdsb::out("edge_count_in", read_graph_data.file_header.edge_count);
        gdsb::out("graph_directed", read_graph_data.file_header.directed);
        gdsb::out("graph_weighted", read_graph_data.file_header.weighted);
        gdsb::out("random_walk_count", random_walk_count);
    }

    benchmark::MemoryFootprintT mem_before_rw = gdsb::memory_usage_in_kb();
    benchmark::MemoryFootprintT mem_before_rw_global = benchmark::mpi_reduce_memory_footprint(mem_before_rw);
    if (benchmark::is_master_process())
    {
        gdsb::out("mem_before_rw_global", mem_before_rw_global);
    }

    switch (rw_algo_param)
    {
        case benchmark::RandomWalkAlgorithmParameter::node2vec:
        {
            Node2vecConf node2vec_conf;
            node2vec_conf.p = node2vec_p;
            node2vec_conf.q = node2vec_q;
            node2vec_conf.walk_length = random_walk_length;
            node2vec_conf.walker_num = random_walk_count;

            if (read_graph_data.file_header.weighted)
            {
                experiment_duration =
                    experiments::vanilla_node2vec<gdsb::WeightedEdges32, real_t>(std::move(read_graph_data.weighted_edges),
                                                                                 read_graph_data.file_header.vertex_count,
                                                                                 node2vec_conf, walk_conf);
            }
            else
            {
                experiment_duration =
                    experiments::vanilla_node2vec<gdsb::Edges32, EmptyData>(std::move(read_graph_data.edges),
                                                                            read_graph_data.file_header.vertex_count,
                                                                            node2vec_conf, walk_conf);
            }
            break;
        }

        case benchmark::RandomWalkAlgorithmParameter::crw:
        {

            if (read_graph_data.file_header.weighted)
            {
                experiment_duration =
                    experiments::crw_vanilla<gdsb::WeightedEdges32, real_t>(std::move(read_graph_data.weighted_edges),
                                                                                 read_graph_data.file_header.vertex_count, walk_conf,
                                                                                 random_walk_length, random_walk_count);
            }
            else
            {
                experiment_duration =
                    experiments::crw_vanilla<gdsb::Edges32, EmptyData>(std::move(read_graph_data.edges),
                                                                            read_graph_data.file_header.vertex_count,
                                                                            walk_conf, random_walk_length, random_walk_count);
            }
            break;
        }

        default:
            // Should not be reachable!
            break;
    }

    benchmark::MemoryFootprintT mem_after_rw = gdsb::memory_usage_in_kb();
    benchmark::MemoryFootprintT mem_after_rw_global = benchmark::mpi_reduce_memory_footprint(mem_after_rw);
    benchmark::MemoryFootprintT mem_after_rw_max = benchmark::mpi_max_memory_footprint(mem_after_rw);
    if (benchmark::is_master_process())
    {
        gdsb::out("mem_after_rw_global", mem_after_rw_global);
        gdsb::out("mem_after_rw_max", mem_after_rw_max);
    }

    // Sadly, in KK we can't really measure the file output time on it's own.
    // Therefore, we simply add this after output mem data point to complete all
    // data points.
    benchmark::MemoryFootprintT mem_after_output = gdsb::memory_usage_in_kb();
    benchmark::MemoryFootprintT mem_after_output_global = benchmark::mpi_reduce_memory_footprint(mem_after_output);
    benchmark::MemoryFootprintT mem_after_output_max = benchmark::mpi_max_memory_footprint(mem_after_output);
    if (benchmark::is_master_process())
    {
        gdsb::out("mem_after_output_global", mem_after_output_global);
        gdsb::out("mem_after_output_max", mem_after_output_max);
        gdsb::out("experiment_duration", experiment_duration.count());
    }

    return 0;
}