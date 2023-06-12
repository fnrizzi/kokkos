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

#ifndef KOKKOS_STD_ALGORITHMS_ROTATE_COPY_HPP
#define KOKKOS_STD_ALGORITHMS_ROTATE_COPY_HPP

#include "impl/Kokkos_RotateCopy.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 OutputIterator>
rotate_copy(const ExecutionSpace& ex, InputIterator first,
            InputIterator n_first, InputIterator last, OutputIterator d_first) {
  return Impl::rotate_copy_exespace_impl(
      "Kokkos::rotate_copy_iterator_api_default", ex, first, n_first, last,
      d_first);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 OutputIterator>
rotate_copy(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator n_first, InputIterator last,
            OutputIterator d_first) {
  return Impl::rotate_copy_exespace_impl(label, ex, first, n_first, last,
                                         d_first);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto rotate_copy(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::rotate_copy_exespace_impl(
      "Kokkos::rotate_copy_view_api_default", ex, cbegin(source),
      cbegin(source) + n_location, cend(source), begin(dest));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto rotate_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::rotate_copy_exespace_impl(label, ex, cbegin(source),
                                         cbegin(source) + n_location,
                                         cend(source), begin(dest));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class InputIterator, class OutputIterator>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, OutputIterator>
rotate_copy(const TeamHandleType& teamHandle, InputIterator first,
            InputIterator n_first, InputIterator last, OutputIterator d_first) {
  return Impl::rotate_copy_team_impl(teamHandle, first, n_first, last, d_first);
}

template <
    class TeamHandleType, class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto rotate_copy(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source,
    std::size_t n_location,
    const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::rotate_copy_team_impl(teamHandle, cbegin(source),
                                     cbegin(source) + n_location, cend(source),
                                     begin(dest));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
