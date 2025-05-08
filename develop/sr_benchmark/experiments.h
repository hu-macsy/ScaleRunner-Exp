#pragma once

#include <scalerunner/first_order_rw.h>
#include <scalerunner/graph.h>
#include <scalerunner/mpi_utils.h>
#include <scalerunner/node2vec.h>
#include <scalerunner/random_walk.h>
#include <scalerunner/rw_scheduler.h>
#include <scalerunner/second_order_rw.h>
#include <scalerunner/utils.h>

#include <atomic>
#include <vector>

struct Node2VecData
{
    sr::ConsecutivePaths& paths;
    sr::node2vec::Parameter& parameter;
};

namespace unweighted
{

std::chrono::milliseconds
crw(sr::ConsecutivePaths&, dhb::Matrix<sr::Weight> const&, uint64_t random_walk_length = 50, uint64_t random_walk_count = 0);

std::chrono::milliseconds personal_page_rank(sr::ConsecutivePaths&,
                                             dhb::Matrix<sr::Weight> const&,
                                             uint64_t random_walk_length = 50,
                                             uint64_t random_walk_count = 0);

std::chrono::milliseconds node2vec_rejection_sampling(Node2VecData&,
                                                      dhb::Matrix<sr::Weight> const& graph,
                                                      uint64_t random_walk_length = 50,
                                                      uint64_t random_walk_count = 0);


} // namespace unweighted

namespace weighted
{
std::chrono::milliseconds
crw(sr::ConsecutivePaths&, dhb::Matrix<sr::Weight> const&, uint64_t random_walk_length = 50, uint64_t random_walk_count = 0);

std::chrono::milliseconds personal_page_rank(sr::ConsecutivePaths&,
                                             dhb::Matrix<sr::Weight> const&,
                                             uint64_t random_walk_length = 50,
                                             uint64_t random_walk_count = 0);


std::chrono::milliseconds node2vec_rejection_sampling(Node2VecData&,
                                                      dhb::Matrix<sr::Weight> const& graph,
                                                      uint64_t random_walk_length = 50,
                                                      uint64_t random_walk_count = 0);

} // namespace weighted
