#include "../kk_benchmark/helpers.h"
#include "experiments.h"
#include "mpi_helper.h"

#include <scalerunner/mpi_utils.h>

#include <gdsb/experiment.h>
#include <gdsb/graph.h>
#include <gdsb/graph_input.h>
#include <gdsb/mpi_graph_io.h>

#include <dhb/batcher.h>
#include <dhb/dynamic_hashed_blocks.h>

#include <CLI/CLI.hpp>

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

void print(dhb::Edges const& edges)
{
    std::cout << "Edges: ";
    for (auto e = std::begin(edges); e != std::end(edges); ++e)
    {
        std::cout << "{" << e->source << ", " << e->target.vertex << " : " << e->target.data.weight << "} ";
    }
    std::cout << std::endl;
}

enum class RWAlgorithm
{
    crw,
    ppr,
    node2vec
};

RWAlgorithm read_rw_algorithm(std::string const& rw_algorithm)
{
    if (rw_algorithm == "crw")
    {
        return RWAlgorithm::ppr;
    }

    if (rw_algorithm == "ppr")
    {
        return RWAlgorithm::ppr;
    }

    if (rw_algorithm == "node2vec")
    {
        return RWAlgorithm::node2vec;
    }

    std::cerr << "RW Algorithm [" << rw_algorithm << "] unknown!";
    std::exit(1);
}

sr::node2vec::Parameter make_n2v_parameter(float const p_return, float const q_in_out)
{
    bool const either_is_infinit =
        p_return == std::numeric_limits<float>::infinity() || q_in_out == std::numeric_limits<float>::infinity();
    bool const either_is_nan = std::isnan(p_return) || std::isnan(q_in_out);

    if (either_is_infinit || either_is_nan)
    {
        std::cerr << "At least one of the Node2Vec parameters (p, q) are infinit or not a number." << std::endl;
        std::exit(1);
    }

    return sr::node2vec::Parameter{ p_return, q_in_out };
}

uint64_t read_in_graph(gdsb::BinaryGraphHeader const& header,
                       dhb::Matrix<sr::Weight>& graph,
                       gdsb::mpi::FileWrapper& binary_graph_mpi,
                       uint64_t const max_batch_size)
{
    gdsb::mpi::ReadBatch read_batch;
    read_batch.batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
    uint32_t const cob = gdsb::count_of_batches(header.edge_count, read_batch.batch_size);
    uint64_t total_edge_count = 0;

    if (header.weighted)
    {
        if (header.dynamic)
        {
            gdsb::WeightedTimestampedEdges32 edges;
            gdsb::mpi::MPIWeightedTimestampedEdge32 mpi_edge_t;
            read_batch.edge_size_in_bytes = sizeof(gdsb::WeightedTimestampedEdge32);
            auto cmp = [](gdsb::WeightedTimestampedEdge32 const& a, gdsb::WeightedTimestampedEdge32 const& b)
            { return a.edge.source < b.edge.source; };
            auto key = [](gdsb::WeightedTimestampedEdge32 const& e) { return e.edge.source; };
            auto get_edge_f = [](gdsb::WeightedTimestampedEdge32 const& e) { return e.edge.source; };
            dhb::BatchParallelizer<gdsb::WeightedTimestampedEdge32> par;

            // If the graph is not weighted, then there are no timestamps.
            if (header.directed)
            {
                auto fun = [&](gdsb::WeightedTimestampedEdge32 const& e)
                { graph.neighbors(e.edge.source).insert(e.edge.target.vertex, e.edge.target.weight); };

                for (uint32_t batch = 0; batch < cob; ++batch)
                {
                    auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                    total_edge_count += count;
                    edges.resize(count);

                    read_batch.batch_size = count;
                    gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                           mpi_edge_t.get());

                    par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
                }

                return total_edge_count;
            }
            else
            {
                gdsb::WeightedTimestampedEdges32 back_edges;

                auto fun = [&](gdsb::WeightedTimestampedEdge32 const& e)
                { graph.neighbors(e.edge.source).insert(e.edge.target.vertex, e.edge.target.weight); };

                for (uint32_t batch = 0; batch < cob; ++batch)
                {
                    auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                    total_edge_count += count;
                    edges.resize(count);
                    back_edges.resize(count);

                    read_batch.batch_size = count;
                    gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                           mpi_edge_t.get());

#pragma omp parallel for
                    for (size_t idx = 0u; idx < edges.size(); ++idx)
                    {
                        back_edges[idx].edge.source = edges[idx].edge.target.vertex;
                        back_edges[idx].edge.target.vertex = edges[idx].edge.source;
                        back_edges[idx].edge.target.weight = edges[idx].edge.target.weight;
                    }

                    par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
                    par(back_edges.begin(), back_edges.end(), std::move(get_edge_f), std::move(cmp), fun);
                }

                return total_edge_count;
            }
        }

        // Weighted Graph!
        gdsb::WeightedEdges32 edges;
        gdsb::mpi::MPIWeightedEdge32 mpi_edge_t;
        read_batch.edge_size_in_bytes = sizeof(gdsb::WeightedEdge32);
        auto cmp = [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        auto get_edge_f = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        dhb::BatchParallelizer<gdsb::WeightedEdge32> par;

        // If the graph is not weighted, then there are no timestamps.
        if (header.directed)
        {
            auto fun = [&](gdsb::WeightedEdge32 const& e)
            { graph.neighbors(e.source).insert(e.target.vertex, e.target.weight); };

            for (uint32_t batch = 0; batch < cob; ++batch)
            {
                auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                total_edge_count += count;
                edges.resize(count);

                read_batch.batch_size = count;
                gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                       mpi_edge_t.get());

                par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
            }

            return total_edge_count;
        }
        else
        {
            gdsb::WeightedEdges32 back_edges;

            auto fun = [&](gdsb::WeightedEdge32 const& e)
            { graph.neighbors(e.source).insert(e.target.vertex, e.target.weight); };

            for (uint32_t batch = 0; batch < cob; ++batch)
            {
                auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                total_edge_count += count;
                edges.resize(count);
                back_edges.resize(count);

                read_batch.batch_size = count;
                gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                       mpi_edge_t.get());

#pragma omp parallel for
                for (size_t idx = 0u; idx < edges.size(); ++idx)
                {
                    back_edges[idx].source = edges[idx].target.vertex;
                    back_edges[idx].target.vertex = edges[idx].source;
                    back_edges[idx].target.weight = edges[idx].target.weight;
                }

                par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
                par(back_edges.begin(), back_edges.end(), std::move(get_edge_f), std::move(cmp), fun);
            }

            return total_edge_count;
        }
    }
    else
    {
        // Unweighted Graph!
        gdsb::Edges32 edges;
        gdsb::mpi::MPIEdge32 mpi_edge_t;
        read_batch.edge_size_in_bytes = sizeof(gdsb::Edge32);
        sr::Weight constexpr default_weight = 1.f;
        auto cmp = [](gdsb::Edge32 const& a, gdsb::Edge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::Edge32 const& e) { return e.source; };
        auto get_edge_f = [](gdsb::Edge32 const& e) { return e.source; };
        dhb::BatchParallelizer<gdsb::Edge32> par;

        // If the graph is not weighted, then there are no timestamps.
        if (header.directed)
        {
            auto fun = [&](gdsb::Edge32 const& e) { graph.neighbors(e.source).insert(e.target, default_weight); };

            for (uint32_t batch = 0; batch < cob; ++batch)
            {
                auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                total_edge_count += count;
                edges.resize(count);

                read_batch.batch_size = count;
                gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                       mpi_edge_t.get());

                par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
            }

            return total_edge_count;
        }
        else
        {
            gdsb::Edges32 back_edges;

            auto fun = [&](gdsb::Edge32 const& e) { graph.neighbors(e.source).insert(e.target, default_weight); };

            for (uint32_t batch = 0; batch < cob; ++batch)
            {
                auto const [offset, count] = gdsb::fair_batch_offset(read_batch.batch_size, batch, cob, header.edge_count);
                total_edge_count += count;
                edges.resize(count);
                back_edges.resize(count);

                read_batch.batch_size = count;
                gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch,
                                                       mpi_edge_t.get());

#pragma omp parallel for
                for (size_t idx = 0u; idx < edges.size(); ++idx)
                {
                    back_edges[idx].source = edges[idx].target;
                    back_edges[idx].target = edges[idx].source;
                }

                par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
                par(back_edges.begin(), back_edges.end(), std::move(get_edge_f), std::move(cmp), fun);
            }

            return total_edge_count;
        }
    }
}

//! Quick launch
//! mpiexec -n 2 ./dev-builds/sr_benchmark/bin/sr_benchmark -p instances/aves-barn-swallow-non-physical.data -n 18
//! --write-output -o ./output/aves-barn-swallow-non-physical
//!
//! mpiexec -n 2 ./dev-builds/sr_benchmark/bin/sr_benchmark -p instances/web-Google.data -r node2vec
int main(int argc, char** argv)
{
    mpi_helper::MPI_Instance mpi_instance(&argc, &argv);
    if (sr::mpi::is_root())
    {
        gdsb::out("mpi_rank", mpi_helper::mpi_rank());
        gdsb::out("mpi_size", mpi_helper::mpi_size());
    }

    //======================================================================
    // CLI PARSE
    CLI::App app{ "sr_benchmark" };

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
    std::string random_walk_algorithm = "ppr";
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

    // not used in benchmark but in graphtool:
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

    // Choosing this batch size parameter as this is the best performing
    // according to DHB paper, batch size experiment.
    uint64_t max_batch_size = 1u << 17u;
    app.add_option("--mbs", max_batch_size, "Batch size for DHB parallel insert batch.");

    //======================================================================
    // Parse Input Parameter
    CLI11_PARSE(app, argc, argv);

    auto error_handler_f = [](std::filesystem::path const& experiment_output_directory)
    {
        std::cerr << "Path: [" << experiment_output_directory
                  << "] is neither a directory, nor could a directory with that name be created." << std::endl;

        std::exit(EXIT_FAILURE);
    };
    std::filesystem::path experiment_output_path =
        benchmark::provide_experiment_output_path(std::move(error_handler_f), rw_write_output,
                                                  experiment_output_path_input, experiment_name);

    std::filesystem::path graph_file_path(graph_file_path_in);

    sr::node2vec::Parameter node2vec_parameter = make_n2v_parameter(node2vec_p, node2vec_q);

    if (sr::mpi::is_root())
    {
        gdsb::out("graph_file_path_in", graph_file_path.c_str());
        gdsb::out("graph_name", graph_file_path.filename());
        gdsb::out("graph_category", graph_category);
        gdsb::out("experiment_name", experiment_name);
        gdsb::out("rw_write_output", rw_write_output);
        gdsb::out("experiment_output_path_input", experiment_output_path_input);
        gdsb::out("experiment_output_path", experiment_output_path);
        gdsb::out("random_walk_algorithm", random_walk_algorithm);
        gdsb::out("random_walk_length", random_walk_length);
        gdsb::out("thread_count", omp_get_max_threads());
        gdsb::out("p_return", node2vec_parameter.p_return);
        gdsb::out("q_in_out", node2vec_parameter.q_in_out);
        gdsb::out("max_batch_size", max_batch_size);
    }

    uint64_t const mem_before_graph_read_operation = gdsb::memory_usage_in_kb();
    uint64_t const mem_before_graph_read_operation_global = mpi_helper::mpi_reduce_memory_footprint(mem_before_graph_read_operation);
    if (sr::mpi::is_root())
    {
        gdsb::out("mem_before_graph_read_operation_global", mem_before_graph_read_operation_global);
    }

    gdsb::WallTimer graph_read_timer;
    graph_read_timer.start();
    //======================================================================
    // Graph IO
    gdsb::mpi::FileWrapper binary_graph_mpi{ graph_file_path };
    gdsb::BinaryGraphHeader header = gdsb::mpi::read_binary_graph_header(binary_graph_mpi.get());

    if (sr::mpi::is_root())
    {
        gdsb::out("vertex_count_in", header.vertex_count);
        gdsb::out("edge_count_in", header.edge_count);
        gdsb::out("graph_directed", header.directed);
        gdsb::out("graph_weighted", header.weighted);
        gdsb::out("graph_dynamic", header.dynamic);
    }

    if (header.vertex_id_byte_size != sizeof(dhb::Vertex))
    {
        std::cout << "Vertex ID size in bytes read from file is not equal to expected size: " << header.vertex_id_byte_size
                  << " != " << sizeof(dhb::Vertex) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (header.weighted and header.weight_byte_size > sizeof(sr::Weight))
    {
        std::cout << "Weight size in bytes read from file is not equal to expected size: " << header.weight_byte_size
                  << " != " << sizeof(sr::Weight) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (header.dynamic and header.timestamp_byte_size > sizeof(gdsb::Timestamp32))
    {
        std::cout << "Timestamp size in bytes read from file is not equal to expected size: " << header.weight_byte_size
                  << " != " << sizeof(gdsb::Timestamp32) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    dhb::Matrix<sr::Weight> graph(header.vertex_count);
    // For timestamped edges we use the latest timestamp.
    bool constexpr update_edge = true;
    bool constexpr no_update_edge = false;

    uint64_t const read_in_edge_count = read_in_graph(header, graph, binary_graph_mpi, max_batch_size);

    graph_read_timer.end();
    std::chrono::nanoseconds const graph_read_duration_ns = graph_read_timer.duration();
    std::chrono::milliseconds const graph_read_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(graph_read_duration_ns);

    auto const read_success = read_in_edge_count == header.edge_count;

    if (!read_success)
    {
        std::cerr << "Error while reading binary graph file. Exit Program!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto const inserted_all_edges = graph.edges_count() == ((uint32_t(!header.directed) + 1u) * header.edge_count);

    if (!header.dynamic && !inserted_all_edges)
    {
        std::cerr << "Error while constructing graph data structure: not all edges have been inserted. Exit Program!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (graph.vertices_count() != header.vertex_count)
    {
        std::cerr << "Vertex count of read graph file is not equal to vertex count of DHB graph object: "
                  << graph.vertices_count() << " != " << header.vertex_count << ". Exit program." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    auto rw_count_error_f = []()
    {
        std::cerr << "Can not determine random walk count due to missing random walk count and graph vertex count "
                     "parameters."
                  << std::endl;

        std::exit(EXIT_FAILURE);
    };

    uint64_t const mem_after_graph_read_operation = gdsb::memory_usage_in_kb();
    uint64_t const mem_after_graph_read_operation_global = mpi_helper::mpi_reduce_memory_footprint(mem_after_graph_read_operation);
    uint64_t const mem_after_graph_read_operation_max = mpi_helper::mpi_max_memory_footprint(mem_after_graph_read_operation);
    if (sr::mpi::is_root())
    {
        gdsb::out("graph_read_duration_ms", graph_read_duration_ms.count());
        gdsb::out("mem_after_graph_read_operation_global", mem_after_graph_read_operation_global);
        gdsb::out("mem_after_graph_read_operation_max", mem_after_graph_read_operation_max);
    }

    random_walk_count = benchmark::check_random_walk_count_parameters(std::move(rw_count_error_f), random_walk_count,
                                                                      graph.vertices_count());

    if (sr::mpi::is_root())
    {
        gdsb::out("random_walk_count", random_walk_count);
    }

    uint64_t const local_random_walk_count = [&]()
    {
        uint64_t rw_count = random_walk_count / mpi_helper::mpi_size();

        if (sr::mpi::is_root())
        {
            rw_count += random_walk_count % mpi_helper::mpi_size();
        }

        return rw_count;
    }();

    if (sr::mpi::is_root())
    {
        gdsb::out("local_random_walk_count", local_random_walk_count);
    }

    //======================================================================
    // Experiments
    uint64_t const mem_before_rw = gdsb::memory_usage_in_kb();
    uint64_t const mem_before_rw_global = mpi_helper::mpi_reduce_memory_footprint(mem_before_rw);
    if (sr::mpi::is_root())
    {
        gdsb::out("mem_before_rw_global", mem_before_rw_global);
    }

    sr::ConsecutivePaths paths(random_walk_count, random_walk_length);
    RWAlgorithm rw_algorithm = read_rw_algorithm(random_walk_algorithm);

    std::chrono::milliseconds experiment_duration = [&]() -> std::chrono::milliseconds
    {
        switch (rw_algorithm)
        {
            case RWAlgorithm::crw:
                if (header.weighted)
                {
                    return weighted::crw(paths, graph, random_walk_length, local_random_walk_count);
                }
                else
                {
                    return unweighted::crw(paths, graph, random_walk_length, local_random_walk_count);
                }

            case RWAlgorithm::ppr:
                if (header.weighted)
                {
                    return weighted::personal_page_rank(paths, graph, random_walk_length, local_random_walk_count);
                }
                else
                {
                    return unweighted::personal_page_rank(paths, graph, random_walk_length, local_random_walk_count);
                }

            case RWAlgorithm::node2vec:
            {
                Node2VecData n2v_data{ paths, node2vec_parameter };

                if (header.weighted)
                {
                    return weighted::node2vec_rejection_sampling(n2v_data, graph, random_walk_length, local_random_walk_count);
                }
                else
                {
                    return unweighted::node2vec_rejection_sampling(n2v_data, graph, random_walk_length, local_random_walk_count);
                }
            }

            default:
                std::cerr << "Error: Algorithm not implemented!" << std::endl;
                std::exit(1);
        }
    }();

    uint64_t const mem_after_rw = gdsb::memory_usage_in_kb();
    uint64_t const mem_after_rw_global = mpi_helper::mpi_reduce_memory_footprint(mem_after_rw);
    uint64_t const mem_after_rw_max = mpi_helper::mpi_max_memory_footprint(mem_after_rw);
    if (sr::mpi::is_root())
    {
        gdsb::out("mem_after_rw_global", mem_after_rw_global);
        gdsb::out("mem_after_rw_max", mem_after_rw_max);
    }

    if (sr::mpi::is_root())
    {
        gdsb::out("experiment_duration", experiment_duration.count());
    }

    std::chrono::milliseconds output_duration{ 0 };

    if (rw_write_output)
    {
        auto mpi_output_f = [&]()
        {
            gdsb::mpi::FileWrapper file(experiment_output_path, true, 0, MPI_MODE_CREATE | MPI_MODE_WRONLY);
            sr::mpi::stream(file.get(), paths);
            return true;
        };

        output_duration = gdsb::benchmark(std::move(mpi_output_f));
    }

    if (sr::mpi::is_root())
    {
        gdsb::out("output_duration", output_duration.count());
    }

    uint64_t const mem_after_output = gdsb::memory_usage_in_kb();
    uint64_t const mem_after_output_global = mpi_helper::mpi_reduce_memory_footprint(mem_after_output);
    uint64_t const mem_after_outputmax = mpi_helper::mpi_max_memory_footprint(mem_after_output);
    if (sr::mpi::is_root())
    {
        gdsb::out("mem_after_output_global", mem_after_output_global);
        gdsb::out("mem_after_outputmax", mem_after_outputmax);
    }

    return 0;
}