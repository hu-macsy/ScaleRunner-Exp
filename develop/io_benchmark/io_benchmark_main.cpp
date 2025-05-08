#include "../kk_benchmark/helpers.h"
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

uint64_t read_in_graph_mpi(std::filesystem::path const& graph_file_path)
{
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

    if (header.dynamic and header.timestamp_byte_size > sizeof(gdsb::Timestamp32))
    {
        std::cout << "Timestamp size in bytes read from file is not equal to expected size: " << header.weight_byte_size
                  << " != " << sizeof(gdsb::Timestamp32) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    gdsb::mpi::ReadBatch read_batch;
    read_batch.batch_size = header.edge_count;

    if (header.weighted)
    {
        if (header.dynamic)
        {
            gdsb::WeightedTimestampedEdges32 edges;
            gdsb::mpi::MPIWeightedTimestampedEdge32 mpi_edge_t;
            read_batch.edge_size_in_bytes = sizeof(gdsb::WeightedTimestampedEdge32);

            // If the graph is not weighted, then there are no timestamps.
            // Direction does not matter as we we would only insert the back
            // edges which does not matter for the io benchmark.
            edges.resize(read_batch.batch_size);

            gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch, mpi_edge_t.get());

            return edges.size();
        }

        // Weighted Graph!
        gdsb::WeightedEdges32 edges;
        gdsb::mpi::MPIWeightedEdge32 mpi_edge_t;
        read_batch.edge_size_in_bytes = sizeof(gdsb::WeightedEdge32);

        // If the graph is not weighted, then there are no timestamps.
        // If the graph is not weighted, then there are no timestamps.
        // Direction does not matter as we we would only insert the back
        // edges which does not matter for the io benchmark.
        edges.resize(read_batch.batch_size);

        gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch, mpi_edge_t.get());

        return edges.size();
    }
    else
    {
        // Unweighted Graph!
        gdsb::Edges32 edges;
        gdsb::mpi::MPIEdge32 mpi_edge_t;
        read_batch.count_read_in_edges = sizeof(gdsb::Edge32);

        // If the graph is not weighted, then there are no timestamps.
        // If the graph is not weighted, then there are no timestamps.
        // Direction does not matter as we we would only insert the back
        // edges which does not matter for the io benchmark.
        edges.resize(read_batch.batch_size);

        gdsb::mpi::all_read_binary_graph_batch(binary_graph_mpi.get(), header, &(edges[0]), read_batch, mpi_edge_t.get());

        return edges.size();
    }
}

uint64_t read_in_graph_origin(std::filesystem::path const& binary_graph_file,
                              std::filesystem::path const& original_file,
                              std::string const& graph_file_type)
{
    std::ifstream binary_input(binary_graph_file);
    gdsb::BinaryGraphHeader header = gdsb::read_binary_graph_header(binary_input);
    std::ifstream original_file_input(original_file);

    if (sr::mpi::is_root())
    {
        gdsb::out("vertex_count_in", header.vertex_count);
        gdsb::out("edge_count_in", header.edge_count);
        gdsb::out("graph_directed", header.directed);
        gdsb::out("graph_weighted", header.weighted);
        gdsb::out("graph_dynamic", header.dynamic);
    }

    if (header.edge_count > std::numeric_limits<int>::max())
    {
        std::cout << "We can't load a graph that exceeds the MPI counter limit of int trying to load [" + header.edge_count
                  << "] edges. Exit program." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (header.vertex_id_byte_size != sizeof(dhb::Vertex))
    {
        std::cout << "Vertex ID size in bytes read from file is not equal to expected size: " << header.vertex_id_byte_size
                  << " != " << sizeof(dhb::Vertex) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (header.dynamic and header.timestamp_byte_size > sizeof(gdsb::Timestamp32))
    {
        std::cout << "Timestamp size in bytes read from file is not equal to expected size: " << header.weight_byte_size
                  << " != " << sizeof(gdsb::Timestamp32) << ". Exit program" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (header.weighted)
    {
        if (header.dynamic)
        {
            gdsb::WeightedTimestampedEdges32 edges;
            uint64_t total_edge_count = 0;

            // If the graph is not weighted, then there are no timestamps.
            // Direction does not matter as we we would only insert the back
            // edges which does not matter for the io benchmark.
            edges.resize(header.edge_count);

            size_t index = 0;
            gdsb::WeightedTimestampedEdge32 edge;
            auto emplace = [&](gdsb::Vertex32 u, gdsb::Vertex32 v, float w, gdsb::Timestamp32 t)
            { edges[index++] = gdsb::WeightedTimestampedEdge32{ gdsb::WeightedEdge32{ u, v, w }, t }; };

            if (graph_file_type == "edges")
            {
                gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::EdgeListDirectedWeightedNoLoopDynamic, gdsb::Timestamp32>(
                    original_file_input, std::move(emplace));
            }
            else
            {
                gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::MatrixMarketDirectedWeightedNoLoopDynamic, gdsb::Timestamp32>(
                    original_file_input, std::move(emplace));
            }

            return edges.size();
        }

        gdsb::WeightedEdges32 edges;
        uint64_t total_edge_count = 0;

        // If the graph is not weighted, then there are no timestamps.
        // Direction does not matter as we we would only insert the back
        // edges which does not matter for the io benchmark.
        edges.resize(header.edge_count);

        size_t index = 0;
        gdsb::WeightedTimestampedEdge32 edge;
        auto emplace = [&](gdsb::Vertex32 u, gdsb::Vertex32 v, float w)
        { edges[index++] = gdsb::WeightedEdge32{ u, v, w }; };

        if (graph_file_type == "edges")
        {
            gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::EdgeListDirectedWeightedNoLoopStatic>(original_file_input,
                                                                                                            std::move(emplace));
        }
        else
        {
            gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::MatrixMarketDirectedWeightedNoLoopStatic>(original_file_input,
                                                                                                                std::move(emplace));
        }

        return edges.size();
    }
    else
    {
        // Unweighted Graph!
        gdsb::Edges32 edges;
        uint64_t total_edge_count = 0;

        // If the graph is not weighted, then there are no timestamps.
        // Direction does not matter as we we would only insert the back
        // edges which does not matter for the io benchmark.
        edges.resize(header.edge_count);

        size_t index = 0;
        gdsb::WeightedTimestampedEdge32 edge;
        auto emplace = [&](gdsb::Vertex32 u, gdsb::Vertex32 v) { edges[index++] = gdsb::Edge32{ u, v }; };

        if (graph_file_type == "edges")
        {
            gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::EdgeListDirectedUnweightedNoLoopStatic>(original_file_input,
                                                                                                              std::move(emplace));
        }
        else
        {
            gdsb::read_graph<gdsb::Vertex32, decltype(emplace), gdsb::MatrixMarketDirectedUnweightedNoLoopStatic>(original_file_input,
                                                                                                                  std::move(emplace));
        }

        return edges.size();
    }
}

//! Quick launch
//! mpiexec -n 2 ./dev-builds/io_benchmark/bin/io_benchmark -p instances/aves-barn-swallow-non-physical.data
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
    CLI::App app{ "io_benchmark" };

    // Graph Input Parameters
    std::string graph_file_path_in = "";
    app.add_option("-p,--path", graph_file_path_in, "Full path to graph file.")->required()->check(CLI::ExistingFile);
    uint64_t graph_vertex_count = 0;
    app.add_option("-n, --vertex-count", graph_vertex_count, "Provided count of vertices for given graph.");

    std::string experiment_name = "none";
    app.add_option("-e, --experiment", experiment_name, "Name of the experiment for post processing.");

    bool use_gdsb_mpi_io = false;
    app.add_flag("--gdsb-mpi-io", use_gdsb_mpi_io, "Use GDSB MPI I/O.");

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

    //======================================================================
    // Parse Input Parameter
    CLI11_PARSE(app, argc, argv);

    std::filesystem::path graph_file_path(graph_file_path_in);

    std::filesystem::path original_file = graph_file_path.parent_path() / graph_file_path.filename().stem();

    if (sr::mpi::is_root())
    {
        gdsb::out("graph_file_path_in", graph_file_path.c_str());
        gdsb::out("graph_name", graph_file_path.filename());
        gdsb::out("original_file_path", original_file.c_str());
        gdsb::out("graph_category", graph_category);
        gdsb::out("experiment_name", experiment_name);
        gdsb::out("thread_count", omp_get_max_threads());
        gdsb::out("use_gdsb_mpi_io", use_gdsb_mpi_io);
    }

    uint64_t const mem_before_graph_read_operation = gdsb::memory_usage_in_kb();
    uint64_t const mem_before_graph_read_operation_global = mpi_helper::mpi_reduce_memory_footprint(mem_before_graph_read_operation);
    if (sr::mpi::is_root())
    {
        gdsb::out("mem_before_graph_read_operation_global", mem_before_graph_read_operation_global);
    }

    gdsb::WallTimer graph_read_timer;
    graph_read_timer.start();

    uint64_t const read_in_edge_count = [&]()
    {
        if (use_gdsb_mpi_io)
        {
            return read_in_graph_mpi(graph_file_path);
        }

        // return uint64_t(0);
        return read_in_graph_origin(graph_file_path, original_file, graph_file_type);
    }();


    graph_read_timer.end();
    std::chrono::nanoseconds const graph_read_duration_ns = graph_read_timer.duration();
    std::chrono::milliseconds const graph_read_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(graph_read_duration_ns);

    if (sr::mpi::is_root())
    {
        gdsb::out("read_in_edge_count", read_in_edge_count);
    }

    uint64_t const mem_after_graph_read_operation = gdsb::memory_usage_in_kb();
    uint64_t const mem_after_graph_read_operation_global = mpi_helper::mpi_reduce_memory_footprint(mem_after_graph_read_operation);
    uint64_t const mem_after_graph_read_operation_max = mpi_helper::mpi_max_memory_footprint(mem_after_graph_read_operation);
    if (sr::mpi::is_root())
    {
        gdsb::out("graph_read_duration_ms", graph_read_duration_ms.count());
        gdsb::out("mem_after_graph_read_operation_global", mem_after_graph_read_operation_global);
        gdsb::out("mem_after_graph_read_operation_max", mem_after_graph_read_operation_max);
    }

    return 0;
}