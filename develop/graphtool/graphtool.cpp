#include <gdsb/graph_input.h>
#include <gdsb/graph_io_parameters.h>
#include <gdsb/graph_output.h>

#include <CLI/CLI.hpp>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <type_traits>
#include <utility>

int main(int argc, char** argv)
{
    CLI::App app{ "graphtool" };

    //! Graph IO parameters
    std::string graph_file_input_path = "";
    app.add_option("-i,--input", graph_file_input_path, "Full path to input graph file.")->required()->check(CLI::ExistingFile);
    std::string graph_file_output_path = "";
    // TODO: check if this is a valid path..
    app.add_option("-o,--output", graph_file_output_path, "Full path to output graph directory.");

    //! Graph properties
    bool graph_weighted = false;
    app.add_flag("-w, --weighted", graph_weighted, "Graph file contains weight.");
    bool graph_directed = false;
    app.add_flag("-d, --directed", graph_directed, "Flag declaring graph file is directed.");
    bool graph_timestamped = false;
    app.add_flag("-t, --timestamped", graph_timestamped, "Flag declaring graph file carries timestamps.");

    //! Graph File Parameters
    std::string graph_file_type_input = "edges";
    app.add_option("--file-type", graph_file_type_input, "Graph file type with edge list being <edges> and matrix market being <mtx>");

    CLI11_PARSE(app, argc, argv);

    if (graph_timestamped and not graph_weighted)
    {
        std::cerr << "!!! Graph has timestamps but no weights. This converter does not convert unweighted, timestamped "
                     "graphs. !!!"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::filesystem::path graph_path(std::move(graph_file_input_path));
    std::ifstream graph_stream(graph_path);

    gdsb::FileType const graph_file_type = [](std::string const& graph_file_type_input)
    {
        if (graph_file_type_input == "edges")
        {
            return gdsb::FileType::edge_list;
        }
        else if (graph_file_type_input == "mtx")
        {
            return gdsb::FileType::matrix_market;
        }

        std::cerr << "!!! Extension: [" << graph_file_type_input << "] unknown. Exiting program! !!!" << std::endl;
        std::exit(EXIT_FAILURE);
    }(graph_file_type_input);

    gdsb::Edges32 edges;
    gdsb::WeightedEdges32 weighted_edges;
    gdsb::WeightedTimestampedEdges32 timestamped_edges;

    auto emplace_f = [&](gdsb::Vertex32 u, gdsb::Vertex32 v) { edges.push_back(gdsb::Edge32{ u, v }); };

    auto emplace_weighted_f = [&](gdsb::Vertex32 u, gdsb::Vertex32 v, gdsb::Weight w)
    { weighted_edges.push_back(gdsb::WeightedEdge32{ u, { v, w } }); };

    auto emplace_weighted_timestamped_f = [&](gdsb::Vertex32 u, gdsb::Vertex32 v, gdsb::Weight w, gdsb::Timestamp32 t)
    { timestamped_edges.push_back({ { u, { v, w } }, t }); };

    auto [vertex_count, edge_count] = [&]()
    {
        if (graph_weighted)
        {
            if (graph_timestamped)
            {
                switch (graph_file_type)
                {
                    case gdsb::FileType::matrix_market:
                        return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_weighted_timestamped_f), gdsb::MatrixMarketDirectedWeightedNoLoopDynamic>(
                            graph_stream, std::move(emplace_weighted_timestamped_f));
                    case gdsb::FileType::edge_list: // intentional fallthrough
                    default:
                        return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_weighted_timestamped_f), gdsb::EdgeListDirectedWeightedNoLoopDynamic>(
                            graph_stream, std::move(emplace_weighted_timestamped_f));
                }
            }

            switch (graph_file_type)
            {
                case gdsb::FileType::matrix_market:
                    return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_weighted_f), gdsb::MatrixMarketDirectedWeightedNoLoopStatic>(
                        graph_stream, std::move(emplace_weighted_f));
                case gdsb::FileType::edge_list: // intentional fallthrough
                default:
                    return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_weighted_f), gdsb::EdgeListDirectedWeightedNoLoopStatic>(
                        graph_stream, std::move(emplace_weighted_f));
            }
        }

        switch (graph_file_type)
        {
            case gdsb::FileType::matrix_market:
                return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_f), gdsb::MatrixMarketDirectedUnweightedNoLoopStatic>(
                    graph_stream, std::move(emplace_f));
            case gdsb::FileType::edge_list: // intentional fallthrough
            default:
                return gdsb::read_graph<gdsb::Vertex32, decltype(emplace_f), gdsb::EdgeListDirectedUnweightedNoLoopStatic>(
                    graph_stream, std::move(emplace_f));
        }
    }();

    // Making sure the read data is correct, in case of a graph with temporal
    // information, this does not apply.
    bool const is_not_weighted_edge_count_incorrect = not graph_weighted and (edges.size() != edge_count);
    bool const is_weighted_edge_count_incorrect = graph_weighted and (weighted_edges.size() != edge_count);
    bool const edge_count_incorrect = is_not_weighted_edge_count_incorrect or is_weighted_edge_count_incorrect;
    if (not graph_timestamped and edge_count_incorrect)
    {
        std::cerr << "Graph [" << graph_path.filename() << "] -> directed [" << graph_directed << "], weighted: ["
                  << graph_weighted << "], timestamped: [" << graph_timestamped << "]" << std::endl;

        std::cerr << "!!! Discrepancy reading graph data: Edge count is [" << edge_count << "] but read edges count is [";

        if (graph_weighted)
        {
            std::cerr << weighted_edges.size();
        }
        else
        {
            std::cerr << edges.size();
        }

        std::cerr << "]. !!!" << std::endl;
        std::cerr << "!!! Faulty converted graph: [" << graph_path.filename() << "] will be kept on disk !!!" << std::endl;
    }

    if (graph_timestamped)
    {
        // We sort all edges by timestamp. This way we keep order in streamed
        // updates (coming from disk).
        auto compare_f = [](gdsb::WeightedTimestampedEdge32 const& left, gdsb::WeightedTimestampedEdge32 const& right)
        { return left.timestamp < right.timestamp; };

        std::sort(std::begin(timestamped_edges), std::end(timestamped_edges), std::move(compare_f));
    }

    std::cout << "Read vertex count: " << vertex_count << ", edge count: " << edge_count << std::endl;

    std::filesystem::path graph_output_path(std::move(graph_file_output_path));
    std::cout << "Writing file to: " << graph_output_path.c_str() << std::endl;

    std::ofstream binary_file_stream = gdsb::open_binary_file(graph_output_path);

    if (graph_weighted)
    {
        if (graph_timestamped)
        {
            auto write_edge_weighted_timestamped_f = [](std::ofstream& o, gdsb::WeightedTimestampedEdge32 const& edge)
            {
                o.write(reinterpret_cast<const char*>(&edge.edge.source), sizeof(gdsb::Vertex32));
                o.write(reinterpret_cast<const char*>(&edge.edge.target.vertex), sizeof(gdsb::Vertex32));
                o.write(reinterpret_cast<const char*>(&edge.edge.target.weight), sizeof(gdsb::Weight));
                o.write(reinterpret_cast<const char*>(&edge.timestamp), sizeof(gdsb::Timestamp32));
            };

            if (graph_directed)
            {
                gdsb::write_graph<gdsb::BinaryDirectedWeightedDynamic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                    binary_file_stream, timestamped_edges, vertex_count, edge_count, std::move(write_edge_weighted_timestamped_f));
            }
            else
            {
                gdsb::write_graph<gdsb::BinaryUndirectedWeightedDynamic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                    binary_file_stream, timestamped_edges, vertex_count, edge_count, std::move(write_edge_weighted_timestamped_f));
            }
        }
        else
        {
            auto write_edge_weighted_f = [](std::ofstream& o, auto edge)
            {
                o.write(reinterpret_cast<const char*>(&edge.source), sizeof(gdsb::Vertex32));
                o.write(reinterpret_cast<const char*>(&edge.target.vertex), sizeof(gdsb::Vertex32));
                o.write(reinterpret_cast<const char*>(&edge.target.weight), sizeof(gdsb::Weight));
            };

            if (graph_directed)
            {
                gdsb::write_graph<gdsb::BinaryDirectedWeightedStatic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                    binary_file_stream, weighted_edges, vertex_count, edge_count, std::move(write_edge_weighted_f));
            }
            else
            {
                gdsb::write_graph<gdsb::BinaryUndirectedWeightedStatic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                    binary_file_stream, weighted_edges, vertex_count, edge_count, std::move(write_edge_weighted_f));
            }
        }
    }
    else
    {
        auto write_edge_f = [](std::ofstream& o, auto edge)
        {
            o.write(reinterpret_cast<const char*>(&edge.source), sizeof(gdsb::Vertex32));
            o.write(reinterpret_cast<const char*>(&edge.target), sizeof(gdsb::Vertex32));
        };

        if (graph_directed)
        {
            gdsb::write_graph<gdsb::BinaryDirectedUnweightedStatic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                binary_file_stream, edges, vertex_count, edge_count, std::move(write_edge_f));
        }
        else
        {
            gdsb::write_graph<gdsb::BinaryUndirectedUnweightedStatic, gdsb::Vertex32, gdsb::Weight, gdsb::Timestamp32>(
                binary_file_stream, edges, vertex_count, edge_count, std::move(write_edge_f));
        }
    }

    std::cout << "*** Graph [" << graph_path.filename() << "] successfully converted ***" << std::endl;

    return 0;
}
