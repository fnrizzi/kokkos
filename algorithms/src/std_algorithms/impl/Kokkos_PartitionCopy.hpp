//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_STD_ALGORITHMS_PARTITION_COPY_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_PARTITION_COPY_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ValueType>
struct StdPartitionCopyScalar {
  ValueType true_count_;
  ValueType false_count_;
};

template <class FirstFrom, class FirstDestTrue, class FirstDestFalse,
          class PredType>
struct StdPartitionCopyFunctor {
  using index_type = typename FirstFrom::difference_type;
  using value_type = StdPartitionCopyScalar<index_type>;

  FirstFrom m_first_from;
  FirstDestTrue m_first_dest_true;
  FirstDestFalse m_first_dest_false;
  PredType m_pred;

  KOKKOS_FUNCTION
  StdPartitionCopyFunctor(FirstFrom first_from, FirstDestTrue first_dest_true,
                          FirstDestFalse first_dest_false, PredType pred)
      : m_first_from(std::move(first_from)),
        m_first_dest_true(std::move(first_dest_true)),
        m_first_dest_false(std::move(first_dest_false)),
        m_pred(std::move(pred)) {}

  KOKKOS_FUNCTION
  void operator()(const index_type i, value_type& update,
                  const bool final_pass) const {
    const auto& myval = m_first_from[i];
    if (final_pass) {
      if (m_pred(myval)) {
        m_first_dest_true[update.true_count_] = myval;
      } else {
        m_first_dest_false[update.false_count_] = myval;
      }
    }

    if (m_pred(myval)) {
      update.true_count_ += 1;
    } else {
      update.false_count_ += 1;
    }
  }

  KOKKOS_FUNCTION
  void init(value_type& update) const {
    update.true_count_  = 0;
    update.false_count_ = 0;
  }

  KOKKOS_FUNCTION
  void join(value_type& update, const value_type& input) const {
    update.true_count_ += input.true_count_;
    update.false_count_ += input.false_count_;
  }
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                             InputIteratorType from_first,
                             InputIteratorType from_last,
                             OutputIteratorTrueType to_first_true,
                             OutputIteratorFalseType to_first_false,
                             PredicateType pred) {
  // impl uses a scan, this is similar how we implemented copy_if

  // checks
  Impl::static_assert_random_access_and_accessible(
      ex, from_first, to_first_true, to_first_false);
  Impl::static_assert_iterators_have_matching_difference_type(
      from_first, to_first_true, to_first_false);
  Impl::expect_valid_range(from_first, from_last);

  if (from_first == from_last) {
    return {to_first_true, to_first_false};
  }

  using func_type =
      StdPartitionCopyFunctor<InputIteratorType, OutputIteratorTrueType,
                              OutputIteratorFalseType, PredicateType>;

  // run
  const auto num_elements =
      Kokkos::Experimental::distance(from_first, from_last);
  typename func_type::value_type counts{0, 0};
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(from_first, to_first_true, to_first_false, pred), counts);

  // fence not needed here because of the scan into counts

  return {to_first_true + counts.true_count_,
          to_first_false + counts.false_count_};
}

template <class TeamHandleType, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
KOKKOS_FUNCTION ::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy_team_impl(const TeamHandleType& teamHandle,
                         InputIteratorType from_first,
                         InputIteratorType from_last,
                         OutputIteratorTrueType to_first_true,
                         OutputIteratorFalseType to_first_false,
                         PredicateType pred) {
  // impl uses a scan, this is similar how we implemented copy_if

  // checks
  Impl::static_assert_random_access_and_accessible(
      teamHandle, from_first, to_first_true, to_first_false);
  Impl::static_assert_iterators_have_matching_difference_type(
      from_first, to_first_true, to_first_false);
  Impl::expect_valid_range(from_first, from_last);

  if (from_first == from_last) {
    return {to_first_true, to_first_false};
  }

  const std::size_t num_elements =
      Kokkos::Experimental::distance(from_first, from_last);

  // FIXME: there is no parallel_scan overload that accepts TeamThreadRange and
  // return_value, so temporarily serial implementation is used
  std::size_t countTrue  = 0;
  std::size_t countFalse = 0;
  Kokkos::View<std::size_t, typename TeamHandleType::execution_space>
      countTrueView(&countTrue);
  Kokkos::View<std::size_t, typename TeamHandleType::execution_space>
      countFalseView(&countFalse);
  Kokkos::single(Kokkos::PerTeam(teamHandle), [=]() {
    for (std::size_t i = 0; i < num_elements; ++i) {
      const auto& myval = from_first[i];
      if (pred(myval)) {
        to_first_true[countTrueView()++] = myval;
      } else {
        to_first_false[countFalseView()++] = myval;
      }
    }
  });
  teamHandle.team_barrier();

  return {to_first_true + countTrue, to_first_false + countFalse};
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
