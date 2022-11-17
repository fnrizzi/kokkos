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

#ifndef KOKKOS_STD_ALGORITHMS_ADJACENT_FIND_HPP
#define KOKKOS_STD_ALGORITHMS_ADJACENT_FIND_HPP

#include "impl/Kokkos_AdjacentFind.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//

// overload set1
template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const ExecutionSpace& ex, IteratorType first, IteratorType last) {
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const std::string& label, const ExecutionSpace& ex,
              IteratorType first, IteratorType last) {
  return Impl::adjacent_find_exespace_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_view_api_default", ex, KE::begin(v), KE::end(v));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(label, ex, KE::begin(v), KE::end(v));
}

// overload set2
template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const ExecutionSpace& ex, IteratorType first, IteratorType last,
              BinaryPredicateType pred) {
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_iterator_api_default", ex, first, last, pred);
}

template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const std::string& label, const ExecutionSpace& ex,
              IteratorType first, IteratorType last, BinaryPredicateType pred) {
  return Impl::adjacent_find_exespace_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_view_api_default", ex, KE::begin(v), KE::end(v),
      pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(label, ex, KE::begin(v), KE::end(v),
                                           pred);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

// overload set1
template <class TeamHandleType, class IteratorType>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, IteratorType>
adjacent_find(const TeamHandleType& teamHandle, IteratorType first,
              IteratorType last) {
  return Impl::adjacent_find_team_impl(teamHandle, first, last);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto adjacent_find(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_team_impl(teamHandle, KE::begin(v), KE::end(v));
}

// overload set2
template <class TeamHandleType, class IteratorType, class BinaryPredicateType>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, IteratorType>
adjacent_find(const TeamHandleType& teamHandle, IteratorType first,
              IteratorType last, BinaryPredicateType pred) {
  return Impl::adjacent_find_team_impl(teamHandle, first, last, pred);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    class BinaryPredicateType,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto adjacent_find(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v,
    BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_team_impl(teamHandle, KE::begin(v), KE::end(v),
                                       pred);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
