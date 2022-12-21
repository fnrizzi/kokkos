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

#ifndef KOKKOS_STD_ALGORITHMS_GENERATE_N_HPP
#define KOKKOS_STD_ALGORITHMS_GENERATE_N_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_GenerateGenerateN.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class IteratorType, class Size, class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType>
generate_n(const ExecutionSpace& ex, IteratorType first, Size count,
           Generator g) {
  Impl::generate_n_impl("Kokkos::generate_n_iterator_api_default", ex, first,
                        count, std::move(g));
  return first + count;
}

template <class ExecutionSpace, class IteratorType, class Size, class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType>
generate_n(const std::string& label, const ExecutionSpace& ex,
           IteratorType first, Size count, Generator g) {
  Impl::generate_n_impl(label, ex, first, count, std::move(g));
  return first + count;
}

template <class ExecutionSpace, class DataType, class... Properties, class Size,
          class Generator,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto generate_n(const ExecutionSpace& ex,
                const ::Kokkos::View<DataType, Properties...>& view, Size count,
                Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::generate_n_impl("Kokkos::generate_n_view_api_default", ex,
                               begin(view), count, std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties, class Size,
          class Generator,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto generate_n(const std::string& label, const ExecutionSpace& ex,
                const ::Kokkos::View<DataType, Properties...>& view, Size count,
                Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::generate_n_impl(label, ex, begin(view), count, std::move(g));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class IteratorType, class Size, class Generator>
KOKKOS_FUNCTION
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, IteratorType>
    generate_n(const TeamHandleType& teamHandle, IteratorType first, Size count,
               Generator g) {
  Impl::generate_n_team_impl(teamHandle, first, count, std::move(g));
  return first + count;
}

template <
    class TeamHandleType, class DataType, class... Properties, class Size,
    class Generator,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto generate_n(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& view, Size count,
    Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  return Impl::generate_n_team_impl(teamHandle, begin(view), count,
                                    std::move(g));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
