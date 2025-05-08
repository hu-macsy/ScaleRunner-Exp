#pragma once

#include <gdsb/experiment.h>

#include <filesystem>
#include <string>

namespace benchmark
{
using MemoryFootprintT = decltype(gdsb::memory_usage_in_kb());

enum class RandomWalkAlgorithmParameter
{
    node2vec,
    crw
};

constexpr int root_process_rank() { return 0; }

bool is_master_process();

MemoryFootprintT mpi_reduce_memory_footprint(MemoryFootprintT const local_mem_footprint);

MemoryFootprintT mpi_max_memory_footprint(benchmark::MemoryFootprintT local_mem_footprint);

RandomWalkAlgorithmParameter const rw_algorithm(std::string const& random_walk_algorithm);

uint64_t check_random_walk_count_parameters(uint64_t const random_walk_count, uint64_t const graph_vertex_count);

template <class ErrorHandlerF>
std::filesystem::path provide_experiment_output_path(ErrorHandlerF&& error_handler_f,
                                                     bool const rw_write_output,
                                                     std::string const& experiment_output_path_input,
                                                     std::string const& experiment_name)
{
    if (rw_write_output)
    {
        //======================================================================
        // Create output directory
        std::filesystem::path experiment_output_path(experiment_output_path_input);
        experiment_output_path.append(experiment_name);
        std::filesystem::path experiment_output_directory(experiment_output_path.parent_path());
        if (!std::filesystem::is_directory(experiment_output_directory))
        {
            bool const directory_created = std::filesystem::create_directory(experiment_output_directory);
            if (!directory_created && !std::filesystem::is_directory(experiment_output_directory))
            {
                error_handler_f(experiment_output_directory);
            }
        }

        return experiment_output_path;
    }

    return {};
}

template <class ErrorHandlerF>
uint64_t check_random_walk_count_parameters(ErrorHandlerF&& error_handler_f, uint64_t const random_walk_count, uint64_t const graph_vertex_count)
{
    if (random_walk_count == 0)
    {
        if (graph_vertex_count == 0)
        {
            error_handler_f();
        }

        return graph_vertex_count;
    }

    return random_walk_count;
}

} // namespace benchmark