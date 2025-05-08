#include "experiments.h"

#include <scalerunner/first_order_rw.h>
#include <scalerunner/node2vec.h>
#include <scalerunner/random_walk.h>
#include <scalerunner/rw_scheduler.h>
#include <scalerunner/second_order_rw.h>
#include <scalerunner/utils.h>

#include <gdsb/experiment.h>

#include <atomic>
#include <vector>

std::chrono::milliseconds unweighted::personal_page_rank(sr::ConsecutivePaths& path,
                                                         dhb::Matrix<sr::Weight> const& graph,
                                                         uint64_t random_walk_length,
                                                         uint64_t random_walk_count)
{
    constexpr float travel_to_random_vertex_p = 0.1f;
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    auto rwr_condition = [&random_engines](dhb::Matrix<sr::Weight> const& graph) -> std::optional<dhb::Vertex>
    {
        uint32_t const thread_id = omp_get_thread_num();
        float const prob = random_engines[thread_id].f_rng(1.f);
        if (prob <= travel_to_random_vertex_p)
        {
            return random_engines[thread_id].v_rng(dhb::Vertex(graph.vertices_count() - 1));
        }

        return {};
    };

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n)
    {
        auto candidate = rwr_condition(graph);
        if (candidate)
        {
            return candidate;
        }

        uint32_t const thread_id = omp_get_thread_num();
        return sr::first_order::step(n, random_engines[thread_id].i_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    dhb::Vertex start_vertex = 0u;
    auto personal_page_rank = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::first_order::walk(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        path = sr::schedule(std::move(personal_page_rank), std::move(start_vertex_f), random_walk_count,
                            random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}

std::chrono::milliseconds
unweighted::crw(sr::ConsecutivePaths& path, dhb::Matrix<sr::Weight> const& graph, uint64_t random_walk_length, uint64_t random_walk_count)
{
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n)
    {
        uint32_t const thread_id = omp_get_thread_num();
        return sr::first_order::step(n, random_engines[thread_id].i_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    dhb::Vertex start_vertex = 0u;
    auto personal_page_rank = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::first_order::walk(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        path = sr::schedule(std::move(personal_page_rank), std::move(start_vertex_f), random_walk_count,
                            random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}

std::chrono::milliseconds unweighted::node2vec_rejection_sampling(Node2VecData& data,
                                                                  dhb::Matrix<sr::Weight> const& graph,
                                                                  uint64_t random_walk_length,
                                                                  uint64_t random_walk_count)
{
    constexpr float travel_to_random_vertex_p = 0.1f;
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    float const max_alpha_prob_value = sr::node2vec::max_alpha(data.parameter);

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n,
                      std::optional<dhb::Vertex> last_vertex) -> std::optional<dhb::Vertex>
    {
        return sr::node2vec::unweighted::step_rejection_sampling(graph, n, last_vertex, data.parameter, max_alpha_prob_value,
                                                                 random_engines[omp_get_thread_num()].i_rng,
                                                                 random_engines[omp_get_thread_num()].f_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    auto node2vec = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::second_order(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        data.paths = sr::schedule(std::move(node2vec), std::move(start_vertex_f), random_walk_count, random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}

std::chrono::milliseconds
weighted::personal_page_rank(sr::ConsecutivePaths& path, dhb::Matrix<sr::Weight> const& graph, uint64_t random_walk_length, uint64_t random_walk_count)
{
    constexpr float travel_to_random_vertex_p = 0.1f;
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    auto rwr_condition = [&random_engines](dhb::Matrix<sr::Weight> const& graph) -> std::optional<dhb::Vertex>
    {
        uint32_t const thread_id = omp_get_thread_num();
        float const prob = random_engines[thread_id].f_rng(1.f);
        if (prob <= travel_to_random_vertex_p)
        {
            return random_engines[thread_id].v_rng(dhb::Vertex(graph.vertices_count() - 1));
        }

        return {};
    };

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n)
    {
        auto candidate = rwr_condition(graph);
        if (candidate)
        {
            return candidate;
        }

        uint32_t const thread_id = omp_get_thread_num();
        return sr::first_order::step_weighted(n, random_engines[thread_id].i_rng, random_engines[thread_id].f_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    dhb::Vertex start_vertex = 0u;
    auto personal_page_rank = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::first_order::walk(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        path = sr::schedule(std::move(personal_page_rank), std::move(start_vertex_f), random_walk_count,
                            random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}

std::chrono::milliseconds
weighted::crw(sr::ConsecutivePaths& path, dhb::Matrix<sr::Weight> const& graph, uint64_t random_walk_length, uint64_t random_walk_count)
{
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n)
    {
        uint32_t const thread_id = omp_get_thread_num();
        return sr::first_order::step_weighted(n, random_engines[thread_id].i_rng, random_engines[thread_id].f_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    dhb::Vertex start_vertex = 0u;
    auto crw = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::first_order::walk(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        path = sr::schedule(std::move(crw), std::move(start_vertex_f), random_walk_count, random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}

std::chrono::milliseconds
weighted::node2vec_rejection_sampling(Node2VecData& data, dhb::Matrix<sr::Weight> const& graph, uint64_t random_walk_length, uint64_t random_walk_count)
{
    constexpr float travel_to_random_vertex_p = 0.1f;
    std::vector<sr::RandomNumberGenPack> random_engines(omp_get_max_threads());

    auto step_f = [&](dhb::Matrix<sr::Weight>::ConstNeighborView n,
                      std::optional<dhb::Vertex> last_vertex) -> std::optional<dhb::Vertex>
    {
        uint32_t const thread_id = omp_get_thread_num();
        return sr::node2vec::weighted::step_rejection_sampling(graph, n, last_vertex, data.parameter,
                                                               random_engines[thread_id].i_rng,
                                                               random_engines[thread_id].f_rng);
    };

    auto start_vertex_f = [&](size_t const walk_id) { return walk_id % graph.vertices_count(); };

    auto node2vec = [&](dhb::Vertex const start_vertex, sr::Path::iterator begin, sr::RandomNumberGenPack&)
    { sr::second_order(std::move(step_f), graph, start_vertex, begin, random_walk_length); };

    auto bench_f = [&]()
    {
        data.paths = sr::schedule(std::move(node2vec), std::move(start_vertex_f), random_walk_count, random_walk_length, random_engines);
        return true;
    };

    return gdsb::benchmark(std::move(bench_f));
}