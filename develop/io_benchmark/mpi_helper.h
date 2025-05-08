#pragma once

#include <mpi.h>

#include <cassert>

namespace mpi_helper
{

class MPI_Instance
{
public:
    MPI_Instance(int* argc, char*** argv) { MPI_Init(argc, argv); }

    ~MPI_Instance() { MPI_Finalize(); }
};

inline int root_process_rank() { return 0; }

inline int mpi_rank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline int mpi_size()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

inline bool is_master_process() { return mpi_rank() == root_process_rank(); }

inline uint64_t mpi_reduce_memory_footprint(uint64_t const local_mem_footprint)
{
    uint64_t global_mem_footprint = 0;
    MPI_Reduce(&local_mem_footprint, &global_mem_footprint, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, root_process_rank(), MPI_COMM_WORLD);
    return global_mem_footprint;
}

inline uint64_t mpi_max_memory_footprint(uint64_t const local_mem_footprint)
{
    uint64_t global_max_footprint = 0u;
    MPI_Reduce(&local_mem_footprint, &global_max_footprint, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, root_process_rank(), MPI_COMM_WORLD);
    return global_max_footprint;
}

} // namespace mpi_helper