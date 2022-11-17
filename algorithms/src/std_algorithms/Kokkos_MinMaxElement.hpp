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

#ifndef KOKKOS_STD_ALGORITHMS_MINMAX_ELEMENT_HPP
#define KOKKOS_STD_ALGORITHMS_MINMAX_ELEMENT_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_MinMaxMinmaxElement.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class IteratorType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last) {
  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "Kokkos::minmax_element_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(label, ex,
                                                                first, last);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "Kokkos::minmax_element_iterator_api_default", ex, first, last,
      std::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, first, last, std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "Kokkos::minmax_element_view_api_default", ex, begin(v), end(v));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      label, ex, begin(v), end(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "Kokkos::minmax_element_view_api_default", ex, begin(v), end(v),
      std::move(comp));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, begin(v), end(v), std::move(comp));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <
    class TeamHandleType, class IteratorType,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last) {
  return Impl::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle, first,
                                                            last);
}

template <
    class TeamHandleType, class IteratorType, class ComparatorType,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last,
                                    ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(teamHandle);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, first, last, std::move(comp));
}

template <
    class TeamHandleType, class DataType, class... Properties,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle,
                                                            begin(v), end(v));
}

template <
    class TeamHandleType, class DataType, class ComparatorType,
    class... Properties,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v, ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(teamHandle);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, begin(v), end(v), std::move(comp));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
