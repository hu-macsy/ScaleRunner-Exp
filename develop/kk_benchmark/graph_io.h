#pragma once

#include <gdsb/mpi_graph_io.h>
#include <kklib/mpi_helper.hpp>
#include <kklib/type.hpp>

#include <algorithm>
#include <filesystem>

namespace benchmark
{

template <typename Edges> void insert_return_edges(Edges& edges, size_t const original_size)
{
    // Each thread will copy a dedicated range of the read in edges.
#pragma omp parallel
    {
        size_t const edge_begin = gdsb::batch_offset(original_size, omp_get_thread_num(), omp_get_num_threads());
        size_t const edge_end =
            edge_begin + gdsb::partition_batch_count(original_size, omp_get_thread_num(), omp_get_num_threads());

        auto original_it = std::begin(edges);
        std::advance(original_it, edge_begin);
        auto original_end_it = std::begin(edges);
        std::advance(original_end_it, edge_end);

        size_t const copy_begin = original_size + edge_begin;
        size_t const copy_end = original_size + edge_end;

        auto copy_it = std::begin(edges);
        std::advance(copy_it, copy_begin);
        auto copy_end_it = std::begin(edges);
        std::advance(copy_end_it, copy_end);

        for (; original_it < original_end_it && copy_it < copy_end_it; ++original_it, ++copy_it)
        {
            if constexpr (std::is_same_v<Edges, gdsb::WeightedEdges32>)
            {
                copy_it->source = original_it->target.vertex;
                copy_it->target.vertex = original_it->source;
                copy_it->target.weight = original_it->target.weight;
            }
            else
            {
                copy_it->source = original_it->target;
                copy_it->target = original_it->source;
            }
        }
    }
}

template <typename TargetEdge, typename EdgeData>
void copy_edges(gdsb::WeightedEdges32 const& source_edges, TargetEdge* target_edges, uint64_t const partition_edge_count)
{
    for (size_t i = 0; i < partition_edge_count; ++i)
    {
        target_edges[i].src = source_edges[i].source;
        target_edges[i].dst = source_edges[i].target.vertex;

        if constexpr (std::is_same_v<EdgeData, real_t>)
        {
            target_edges[i].data = source_edges[i].target.weight;
        }
    }
}

template <typename TargetEdge, typename EdgeData>
void copy_edges(gdsb::Edges32 const& source_edges, TargetEdge* target_edges, uint64_t const partition_edge_count)
{
    for (size_t i = 0; i < partition_edge_count; ++i)
    {
        target_edges[i].src = source_edges[i].source;
        target_edges[i].dst = source_edges[i].target;
    }
}

struct ReadGraphData
{
    gdsb::Edges32 edges;
    gdsb::WeightedEdges32 weighted_edges;
    gdsb::BinaryGraphHeader file_header;
};

// Edges must be sorted!
void erase_duplicates(gdsb::WeightedEdges32& edges)
{
    auto binary_pred = [](gdsb::WeightedEdge32& a, gdsb::WeightedEdge32& b)
    {
        bool const source_equals = a.source == b.source;
        if (source_equals)
        {
            return a.target.vertex == b.target.vertex;
        }

        return source_equals;
    };

    auto last = std::unique(std::begin(edges), std::end(edges), std::move(binary_pred));

    edges.erase(last, std::end(edges));
}

// TODO: unify these edges type dependent erase_duplicate() functions?
void erase_duplicates(gdsb::Edges32& edges)
{
    auto binary_pred = [](gdsb::Edge32& a, gdsb::Edge32& b)
    {
        bool const source_equals = a.source == b.source;
        if (source_equals)
        {
            return a.target == b.target;
        }

        return source_equals;
    };

    auto last = std::unique(std::begin(edges), std::end(edges), std::move(binary_pred));

    edges.erase(last, std::end(edges));
}

inline ReadGraphData read_in_edges_from_file(std::filesystem::path const& file_path)
{
    gdsb::mpi::FileWrapper mpi_file{ file_path };
    gdsb::BinaryGraphHeader const file_header = gdsb::mpi::read_binary_graph_header(mpi_file.get());

    gdsb::WeightedEdges32 weighted_edges;
    gdsb::Edges32 edges;
    gdsb::WeightedTimestampedEdges32 weighted_timestamped_edges;

    size_t const partition_edge_count = gdsb::partition_batch_count(file_header.edge_count, get_mpi_rank(), get_mpi_size());

    auto const read_success = [&]()
    {
        if (file_header.weighted)
        {
            if (file_header.dynamic)
            {
                weighted_timestamped_edges.resize(partition_edge_count);

                gdsb::mpi::MPIWeightedTimestampedEdge32 mpi_timestamped_edge_t;
                auto [v_count, e_count] =
                    gdsb::mpi::all_read_binary_graph_partition(mpi_file.get(), file_header, &(weighted_timestamped_edges[0]),
                                                               sizeof(gdsb::WeightedTimestampedEdge32),
                                                               mpi_timestamped_edge_t.get(), get_mpi_rank(), get_mpi_size());


                return e_count == partition_edge_count;
            }

            weighted_edges.resize(partition_edge_count);

            gdsb::mpi::MPIWeightedEdge32 mpi_weighted_edge_t;
            auto [v_count, e_count] =
                gdsb::mpi::all_read_binary_graph_partition(mpi_file.get(), file_header, &(weighted_edges[0]),
                                                           sizeof(gdsb::WeightedEdge32), mpi_weighted_edge_t.get(),
                                                           get_mpi_rank(), get_mpi_size());


            return e_count == partition_edge_count;
        }
        else
        {
            edges.resize(partition_edge_count);

            gdsb::mpi::MPIEdge32 mpi_edge_t;
            auto [v_count, e_count] =
                gdsb::mpi::all_read_binary_graph_partition(mpi_file.get(), file_header, &(edges[0]), sizeof(gdsb::Edge32),
                                                           mpi_edge_t.get(), get_mpi_rank(), get_mpi_size());


            return e_count == partition_edge_count;
        }
    }();

    if (!read_success)
    {
        std::cerr << "Error while reading edges for node partition of the binary file. Exit Program!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (file_header.dynamic)
    {
        // not really a nice solution to have to find duplicates like this, but
        // it sufficient for now
        //
        // TODO: perhaps use DHB to update duplicates -> would require 2m memory
        // when writing edges
        if (file_header.weighted)
        {
            // We don't process timestamped edges so we simply move them to weighted edges.
            for (auto e : weighted_timestamped_edges)
            {
                weighted_edges.push_back({ e.edge.source, { e.edge.target.vertex, e.edge.target.weight } });
            }

            weighted_timestamped_edges.clear();

            std::sort(std::begin(weighted_edges), std::end(weighted_edges),
                      [](gdsb::WeightedEdge32 const& a, gdsb::WeightedEdge32 const& b)
                      {
                          bool const a_less_b = a.source < b.source;
                          if (a_less_b)
                          {
                              return a.target.vertex < b.target.vertex;
                          }

                          return a_less_b;
                      });

            erase_duplicates(weighted_edges);
        }
        else
        {
            // We don't process timestamped edges so we simply move them to  edges.
            for (auto e : weighted_timestamped_edges)
            {
                edges.push_back({ e.edge.source, e.edge.target.vertex });
            }

            weighted_timestamped_edges.clear();

            std::sort(std::begin(edges), std::end(edges),
                      [](gdsb::Edge32 const& a, gdsb::Edge32 const& b)
                      {
                          bool const a_less_b = a.source < b.source;
                          if (a_less_b)
                          {
                              return a.target < b.target;
                          }

                          return a_less_b;
                      });

            erase_duplicates(edges);
        }
    }

    if (!file_header.directed)
    {
        if (file_header.weighted)
        {
            size_t const original_size = weighted_edges.size();
            weighted_edges.resize(original_size * 2);
            insert_return_edges(weighted_edges, original_size);
        }
        else
        {
            size_t const original_size = edges.size();
            edges.resize(original_size * 2);
            insert_return_edges(edges, original_size);
        }
    }

    ReadGraphData read_graph_data;
    read_graph_data.file_header = std::move(file_header);
    read_graph_data.edges = std::move(edges);
    read_graph_data.weighted_edges = std::move(weighted_edges);

    return read_graph_data;
}

} // namespace benchmark