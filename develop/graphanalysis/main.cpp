#include <dhb/dynamic_hashed_blocks.h>

#include <gdsb/batcher.h>
#include <gdsb/experiment.h>
#include <gdsb/graph_input.h>
#include <gdsb/timer.h>

#include <CLI/CLI.hpp>

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

namespace fs = std::experimental::filesystem;

struct VertexDegree
{
    gdsb::Vertex64 vertex{ 0 };
    gdsb::Vertex64 degree{ 0 };
};

bool check_category(std::string const& category)
{
    bool category_represented = false;

    category_represented = (category == "social");
    category_represented = category_represented || (category == "biological");
    category_represented = category_represented || (category == "web");
    category_represented = category_represented || (category == "road");
    category_represented = category_represented || (category == "collaboration");
    category_represented = category_represented || (category == "misc");

    return category_represented;
}

// Quick use:
// ./dev-builds/graphanalysis/bin/graphanalysis -n 1965206 -p instances/web-BerkStan
int main(int argc, char** argv)
{
    //======================================================================
    // CLI PARSE START
    CLI::App app{ "graphanalysis" };

    std::string experiment_name = "graphanalysis";
    app.add_option("-e, --experiment", experiment_name, "Name of the experiment for post processing.");

    // Graph input parameters
    std::string graph_path_raw;
    app.add_option("-p, --path", graph_path_raw, "Full path to graph file.")->required()->check(CLI::ExistingFile);
    bool graph_directed = false;
    app.add_flag("-d, --directed", graph_directed, "Flag declaring graph file is directed.");
    bool graph_weighted = false;
    app.add_flag("-w, --weighted", graph_weighted, "Graph file contains weight.");
    bool graph_timestamped = false;
    app.add_flag("-t, --timestamped", graph_timestamped, "Flag declaring graph file carries timestamps.");
    std::string graph_category = "web";
    app.add_option("--graph-category", graph_category, "Specify graph category (e.g. social/biological/web/road/collaboration/misc).");

    // Unused parameters
    uint64_t graph_vertex_count_input = 0;
    app.add_option("-n, --vertex-count", graph_vertex_count_input, "Provided count of vertices for given graph.");
    uint64_t graph_edge_count_input = 0;
    app.add_option("-m, --edge-count", graph_edge_count_input, "Provided count of edges for given graph.");
    std::string graph_file_type = "edges";
    app.add_option("--file-type", graph_file_type, "The graph file format: edges or mtx.");

    // Graph analysis parameters
    uint32_t data_point_count = 100;
    app.add_option("--data-points", data_point_count, "The amount of data points to generate for the degree distribution.");

    // Choosing this batch size parameter as this is the best performing
    // according to DHB paper, batch size experiment.
    uint64_t max_batch_size = 1u << 17u;
    app.add_option("--mbs", max_batch_size, "Batch size for DHB parallel insert batch.");

    CLI11_PARSE(app, argc, argv);
    // CLI PARSE FINISH
    //======================================================================

    gdsb::out("experiment_name", experiment_name);
    gdsb::out("graph_path_raw", graph_path_raw);
    fs::path graph_path(std::move(graph_path_raw));
    gdsb::out("graph_name", graph_path.filename());
    gdsb::out("graph_file_type", graph_file_type);

    gdsb::out("graph_directed", graph_directed);
    gdsb::out("graph_weighted", graph_weighted);
    gdsb::out("graph_timestamped", graph_timestamped);
    gdsb::out("max_batch_size", max_batch_size);

    if (!check_category(graph_category))
    {
        std::cerr << "Graph category: [" << graph_category << "] unknown!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    gdsb::out("graph_category", graph_category);

    gdsb::out("graph_vertex_count_input", graph_vertex_count_input);
    gdsb::out("graph_edge_count_input", graph_edge_count_input);
    gdsb::out("data_point_count", data_point_count);

    // Setting max count of threads
    int thread_count = std::thread::hardware_concurrency();
    omp_set_num_threads(thread_count);
    if (omp_get_max_threads() != thread_count)
    {
        std::cerr << "OMP can not be set to use as many threads as hardware_concurrency() returns." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    gdsb::out("thread_count", omp_get_max_threads());

    //======================================================================
    // Reading in graph file

    gdsb::Vertex64 vertex_count_in = 0;
    size_t edge_count_in = 0;

    std::ifstream binary_graph(graph_path);
    gdsb::BinaryGraphHeader const header = gdsb::read_binary_graph_header(binary_graph);

    gdsb::out("header.directed", header.directed);
    gdsb::out("header.weighted", header.weighted);
    gdsb::out("header.dynamic", header.dynamic);
    gdsb::out("header.edge_count", header.edge_count);

    dhb::Matrix<gdsb::Weight> matrix(header.vertex_count);
    gdsb::Weight constexpr default_weight = 1.f;
    bool constexpr edge_update = true;
    bool constexpr no_edge_update = false;

    auto read_f_unweighted_32 = [&](std::ifstream& input) -> uint64_t
    {
        gdsb::Edges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::Edge32 const& a, gdsb::Edge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::Edge32 const& e) { return e.source; };
        auto fun = [&](gdsb::Edge32 const& e) { matrix.neighbors(e.source).insert(e.target, default_weight); };
        auto get_edge_f = [](gdsb::Edge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::Edge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            edges.resize(count);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                input.read((char*)&(e->source), sizeof(gdsb::Vertex32));
                input.read((char*)&(e->target), sizeof(gdsb::Vertex32));
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
        }

        return total_edge_count;
    };

    auto read_f_unweighted_undirected_32 = [&](std::ifstream& input) -> uint64_t
    {
        gdsb::Edges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::Edge32 const& a, gdsb::Edge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::Edge32 const& e) { return e.source; };
        auto fun = [&](gdsb::Edge32 const& e) { matrix.neighbors(e.source).insert(e.target, default_weight); };
        auto get_edge_f = [](gdsb::Edge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::Edge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            uint64_t undirected_size = count * 2u;
            edges.resize(undirected_size);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                gdsb::Edge32 edge;
                input.read((char*)&(edge.source), sizeof(gdsb::Vertex32));
                input.read((char*)&(edge.target), sizeof(gdsb::Vertex32));

                e->source = edge.source;
                e->target = edge.target;
                ++e;
                ++total_edge_count;
                e->source = edge.target;
                e->target = edge.source;
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
        }

        return total_edge_count;
    };

    auto read_f_weighted_32 = [&](std::ifstream& input) -> uint64_t
    {
        gdsb::WeightedEdges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        auto fun = [&](gdsb::WeightedEdge32 const& e)
        { matrix.neighbors(e.source).insert(e.target.vertex, e.target.weight); };
        auto get_edge_f = [](gdsb::WeightedEdge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::WeightedEdge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            edges.resize(count);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                input.read((char*)&(e->source), sizeof(gdsb::Vertex32));
                input.read((char*)&(e->target), sizeof(gdsb::Vertex32));
                input.read((char*)&(e->target.weight), sizeof(gdsb::Weight));
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
        }

        return total_edge_count;
    };

    auto read_f_weighted_undirected_32 = [&](std::ifstream& input) -> uint64_t
    {
        gdsb::WeightedEdges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        auto fun = [&](gdsb::WeightedEdge32 const& e)
        { matrix.neighbors(e.source).insert(e.target.vertex, e.target.weight); };
        auto get_edge_f = [](gdsb::WeightedEdge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::WeightedEdge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            uint64_t undirected_size = count * 2u;
            edges.resize(undirected_size);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                gdsb::WeightedEdge32 edge;
                input.read((char*)&(edge.source), sizeof(gdsb::Vertex32));
                input.read((char*)&(edge.target.vertex), sizeof(gdsb::Vertex32));
                input.read((char*)&(edge.target.weight), sizeof(gdsb::Weight));

                e->source = edge.source;
                e->target.vertex = edge.target.vertex;
                e->target.weight = edge.target.weight;
                ++e;
                ++total_edge_count;
                e->source = edge.target.vertex;
                e->target.vertex = edge.source;
                e->target.weight = edge.target.weight;
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), fun);
        }

        return total_edge_count;
    };

    auto read_f_weighted_dynamic_32 = [&](std::ifstream& input)
    {
        gdsb::WeightedEdges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        auto insert_operation = [&](gdsb::WeightedEdge32 const& e)
        {
            auto result = matrix.neighbors(e.source).insert(e.target.vertex, e.target.weight);
            if (!std::get<1>(result))
            {
                std::get<0>(result)->data() = e.target.weight;
            }
        };
        auto get_edge_f = [](gdsb::WeightedEdge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::WeightedEdge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            edges.resize(count);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                input.read((char*)&(e->source), sizeof(gdsb::Vertex32));
                input.read((char*)&(e->target), sizeof(gdsb::Vertex32));
                input.read((char*)&(e->target.weight), sizeof(gdsb::Weight));
                gdsb::Timestamp32 ignored_timestamp_value;
                input.read((char*)&ignored_timestamp_value, sizeof(gdsb::Timestamp32));
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), insert_operation);
        }

        return total_edge_count;
    };

    auto read_f_weighted_dynamic_undirected_32 = [&](std::ifstream& input)
    {
        gdsb::WeightedEdges32 edges;
        uint64_t total_edge_count = 0;

        uint64_t const batch_size = gdsb::fair_batch_size(header.edge_count, max_batch_size);
        uint32_t const cob = gdsb::count_of_batches(header.edge_count, batch_size);

        auto cmp = [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b) { return a.source < b.source; };
        auto key = [](gdsb::WeightedEdge32 const& e) { return e.source; };
        auto insert_operation = [&](gdsb::WeightedEdge32 const& e)
        {
            auto result = matrix.neighbors(e.source).insert(e.target.vertex, e.target.weight);
            if (!std::get<1>(result))
            {
                std::get<0>(result)->data() = e.target.weight;
            }
        };
        auto get_edge_f = [](gdsb::WeightedEdge32 const& e) { return e.source; };

        dhb::BatchParallelizer<gdsb::WeightedEdge32> par;

        for (uint32_t batch = 0; batch < cob; ++batch)
        {
            auto const [offset, count] = gdsb::fair_batch_offset(batch_size, batch, cob, header.edge_count);
            uint64_t undirected_size = count * 2u;
            edges.resize(undirected_size);

            for (auto e = std::begin(edges); e != std::end(edges); ++e, ++total_edge_count)
            {
                gdsb::WeightedEdge32 edge;
                input.read((char*)&(edge.source), sizeof(gdsb::Vertex32));
                input.read((char*)&(edge.target.vertex), sizeof(gdsb::Vertex32));
                input.read((char*)&(edge.target.weight), sizeof(gdsb::Weight));
                gdsb::Timestamp32 ignored_timestamp_value;
                input.read((char*)&ignored_timestamp_value, sizeof(gdsb::Timestamp32));

                e->source = edge.source;
                e->target.vertex = edge.target.vertex;
                e->target.weight = edge.target.weight;
                ++e;
                ++total_edge_count;
                e->source = edge.target.vertex;
                e->target.vertex = edge.source;
                e->target.weight = edge.target.weight;
            }

            par(edges.begin(), edges.end(), std::move(get_edge_f), std::move(cmp), insert_operation);
        }

        return total_edge_count;
    };

    gdsb::WallTimer edge_read_in_timer;
    edge_read_in_timer.start();
    // We intentionally ignore timestamps since we are not interested in reading them out.
    uint64_t const edge_count = [&]()
    {
        if (header.directed)
        {
            if (header.weighted)
            {
                if (header.dynamic)
                {
                    return read_f_weighted_dynamic_32(binary_graph);
                }

                return read_f_weighted_32(binary_graph);
            }
            else
            {
                return read_f_unweighted_32(binary_graph);
            }
        }
        else
        {
            if (header.weighted)
            {
                if (header.dynamic)
                {
                    return read_f_weighted_dynamic_undirected_32(binary_graph);
                }

                return read_f_weighted_undirected_32(binary_graph);
            }

            return read_f_unweighted_32(binary_graph);
        }
    }();
    edge_read_in_timer.end();

    gdsb::out("edge_read_in_time_ms",
              std::chrono::duration_cast<std::chrono::milliseconds>(edge_read_in_timer.duration()).count());

    vertex_count_in = header.vertex_count;
    edge_count_in = edge_count;

    // Reading in graph file
    //======================================================================

    gdsb::out("vertex_count_in", vertex_count_in);
    gdsb::out("edge_count_in", edge_count_in);

    gdsb::out("vertices_count_out", matrix.vertices_count());
    gdsb::out("edge_count_out", matrix.edges_count());

    // Calculate vertex max and min degrees
    gdsb::Vertex64 max_degree_vertex = 0u;
    gdsb::Vertex64 min_degree_vertex = 0u;
    gdsb::Vertex64 max_degree = 0;
    gdsb::Vertex64 min_degree = std::numeric_limits<gdsb::Vertex64>::max();
    gdsb::Vertex64 acc_degree = 0;

    std::vector<VertexDegree> const vertex_degrees = [&]()
    {
        std::vector<VertexDegree> vd(matrix.vertices_count(), VertexDegree{});

        matrix.for_nodes(
            [&](gdsb::Vertex64 const u)
            {
                gdsb::Vertex64 const u_deg = matrix.degree(u);
                acc_degree += u_deg;

                if (u_deg > max_degree)
                {
                    max_degree_vertex = u;
                    max_degree = u_deg;
                }
                else if (u_deg < min_degree)
                {
                    min_degree_vertex = u;
                    min_degree = u_deg;
                }

                vd[u] = VertexDegree{ u, u_deg };
            });

        // Calculate vertex degree distribution
        std::sort(vd.begin(), vd.end(), [&](const auto a, const auto b) { return a.degree > b.degree; });

        return vd;
    }();

    double mean_degree = acc_degree / static_cast<double>(matrix.vertices_count());
    mean_degree = std::ceil(mean_degree * 10.0) / 10.0;

    gdsb::out("max_degree", max_degree);
    gdsb::out("max_degree_vertex", max_degree_vertex);

    gdsb::out("min_degree", min_degree);
    gdsb::out("min_degree_vertex", min_degree_vertex);

    gdsb::out("mean_degree", mean_degree);

    auto choose_representative = [](VertexDegree const& a, VertexDegree const& b) -> VertexDegree
    { return std::max(a, b, [](VertexDegree const& a, VertexDegree const& b) { return a.degree < b.degree; }); };

    size_t const vertices_per_chunk = (matrix.vertices_count() + data_point_count - 1) / data_point_count;
    size_t const chunks = (matrix.vertices_count() + vertices_per_chunk - 1) / vertices_per_chunk;
    std::vector<VertexDegree> data_points(data_point_count, VertexDegree{});
    for (size_t chunk = 0; chunk < chunks; ++chunk)
    {
        size_t const offset = chunk * vertices_per_chunk;

        // Each data point is the max degree of the chunk
        VertexDegree chunk_max_degree = vertex_degrees[offset];

        for (size_t i = offset + 1; i < matrix.vertices_count(); ++i)
        {
            chunk_max_degree = choose_representative(chunk_max_degree, vertex_degrees[i]);
        }

        data_points[chunk] = chunk_max_degree;
    }

    gdsb::out("vertex_id_degree_distribution", data_points.begin(), data_points.end(),
              [](VertexDegree const& v) -> gdsb::Vertex64 { return v.vertex; });
    gdsb::out("degree_degree_distribution", data_points.begin(), data_points.end(),
              [](VertexDegree const& v) -> gdsb::Vertex64 { return v.degree; });

    return 0;
}
