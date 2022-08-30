/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_STD_ALGORITHMS_IS_PARTITIONED_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_IS_PARTITIONED_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class ReducerType, class PredicateType>
struct StdIsPartitionedFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  KOKKOS_FUNCTION
  void operator()(const index_type i, red_value_type& redValue) const {
    const auto predicate_value = m_p(m_first[i]);
    constexpr index_type m_red_id_min =
        ::Kokkos::reduction_identity<index_type>::min();
    constexpr index_type m_red_id_max =
        ::Kokkos::reduction_identity<index_type>::max();
    auto rv = predicate_value ? red_value_type{i, m_red_id_min}
                              : red_value_type{m_red_id_max, i};

    m_reducer.join(redValue, rv);
  }

  KOKKOS_FUNCTION
  StdIsPartitionedFunctor(IteratorType first, ReducerType reducer,
                          PredicateType p)
      : m_first(std::move(first)),
        m_reducer(std::move(reducer)),
        m_p(std::move(p)) {}
};

template <class ExecutionSpace, class IteratorType, class PredicateType>
bool is_partitioned_exespace_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, PredicateType pred) {
  // true if all elements in the range [first, last) that satisfy
  // the predicate "pred" appear before all elements that don't.
  // Also returns true if [first, last) is empty.
  // also true if all elements satisfy the predicate.

  // we implement it by finding:
  // - the max location where predicate is true  (max_loc_true)
  // - the min location where predicate is false (min_loc_false)
  // so the range is partitioned if max_loc_true < (min_loc_false)

  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  // trivial case
  if (first == last) {
    return true;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = StdIsPartitioned<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t =
      StdIsPartitionedFunctor<IteratorType, reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),

                            func_t(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  constexpr index_type red_id_min =
      ::Kokkos::reduction_identity<index_type>::min();
  constexpr index_type red_id_max =
      ::Kokkos::reduction_identity<index_type>::max();

  if (red_result.max_loc_true != red_id_max &&
      red_result.min_loc_false != red_id_min) {
    // when the reduction produces nontrivial values
    return red_result.max_loc_true < red_result.min_loc_false;
  } else if (red_result.max_loc_true == red_id_max &&
             red_result.min_loc_false == 0) {
    // this occurs when all values do NOT satisfy
    // the predicate, and this corner case should also be true
    return true;
  } else if (first + red_result.max_loc_true == --last) {
    // this occurs when all values satisfy the predicate,
    // this corner case should also be true
    return true;
  } else {
    return false;
  }
}

template <class TeamHandleType, class IteratorType, class PredicateType>
KOKKOS_FUNCTION bool is_partitioned_team_impl(const TeamHandleType& teamHandle,
                                              IteratorType first,
                                              IteratorType last,
                                              PredicateType pred) {
  /* see exespace impl for the description of the impl */

  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first);
  Impl::expect_valid_range(first, last);

  // trivial case
  if (first == last) {
    return true;
  }

  // aliases
  using index_type           = typename IteratorType::difference_type;
  using reducer_type         = StdIsPartitioned<index_type>;
  using reduction_value_type = typename reducer_type::value_type;
  using func_t =
      StdIsPartitionedFunctor<IteratorType, reducer_type, PredicateType>;

  // run
  reduction_value_type red_result;
  reducer_type reducer(red_result);
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  ::Kokkos::parallel_reduce(TeamThreadRange(teamHandle, 0, num_elements),
                            func_t(first, reducer, pred), reducer);

  // fence not needed because reducing into scalar

  // decide and return
  constexpr index_type red_id_min =
      ::Kokkos::reduction_identity<index_type>::min();
  constexpr index_type red_id_max =
      ::Kokkos::reduction_identity<index_type>::max();

  if (red_result.max_loc_true != red_id_max &&
      red_result.min_loc_false != red_id_min) {
    // when the reduction produces nontrivial values
    return red_result.max_loc_true < red_result.min_loc_false;
  } else if (red_result.max_loc_true == red_id_max &&
             red_result.min_loc_false == 0) {
    // this occurs when all values do NOT satisfy
    // the predicate, and this corner case should also be true
    return true;
  } else if (first + red_result.max_loc_true == --last) {
    // this occurs when all values satisfy the predicate,
    // this corner case should also be true
    return true;
  } else {
    return false;
  }
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
