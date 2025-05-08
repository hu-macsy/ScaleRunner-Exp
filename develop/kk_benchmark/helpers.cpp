#include "helpers.h"

#include <kklib/mpi_helper.hpp>

#include <mpi.h>

#include <cstdlib>

bool benchmark::is_master_process() { return get_mpi_rank() == root_process_rank(); }

benchmark::MemoryFootprintT benchmark::mpi_reduce_memory_footprint(benchmark::MemoryFootprintT const local_mem_footprint)
{
    MemoryFootprintT global_mem_footprint = 0u;
    MPI_Reduce(&local_mem_footprint, &global_mem_footprint, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, root_process_rank(), MPI_COMM_WORLD);
    return global_mem_footprint;
}

benchmark::MemoryFootprintT benchmark::mpi_max_memory_footprint(benchmark::MemoryFootprintT const local_mem_footprint)
{
    MemoryFootprintT global_max_footprint = 0u;
    MPI_Reduce(&local_mem_footprint, &global_max_footprint, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, root_process_rank(), MPI_COMM_WORLD);
    return global_max_footprint;
}

benchmark::RandomWalkAlgorithmParameter const benchmark::rw_algorithm(std::string const& random_walk_algorithm)
{
    if (random_walk_algorithm == "node2vec")
    {
        return benchmark::RandomWalkAlgorithmParameter::node2vec;
    }
    else if (random_walk_algorithm == "crw")
    {
        return benchmark::RandomWalkAlgorithmParameter::crw;
    }
    else
    {
        if (benchmark::is_master_process())
        {
            std::cerr << "Random walk algorithm [" << random_walk_algorithm << "] Unknown!" << std::endl;
        }

        std::exit(EXIT_FAILURE);
    }
}