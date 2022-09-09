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

#ifndef KOKKOS_STD_ALGORITHMS_FIND_FIRST_OF_HPP
#define KOKKOS_STD_ALGORITHMS_FIND_FIRST_OF_HPP

#include "impl/Kokkos_FindFirstOf.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//

// overload set 1: no binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType1>
find_first_of(const ExecutionSpace& ex, IteratorType1 first, IteratorType1 last,
              IteratorType2 s_first, IteratorType2 s_last) {
  return Impl::find_first_of_exespace_impl(
      "Kokkos::find_first_of_iterator_api_default", ex, first, last, s_first,
      s_last);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType1>
find_first_of(const std::string& label, const ExecutionSpace& ex,
              IteratorType1 first, IteratorType1 last, IteratorType2 s_first,
              IteratorType2 s_last) {
  return Impl::find_first_of_exespace_impl(label, ex, first, last, s_first,
                                           s_last);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto find_first_of(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,
                   const ::Kokkos::View<DataType2, Properties2...>& s_view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_exespace_impl(
      "Kokkos::find_first_of_view_api_default", ex, KE::begin(view),
      KE::end(view), KE::begin(s_view), KE::end(s_view));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,
                   const ::Kokkos::View<DataType2, Properties2...>& s_view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_exespace_impl(label, ex, KE::begin(view),
                                           KE::end(view), KE::begin(s_view),
                                           KE::end(s_view));
}

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType1>
find_first_of(const ExecutionSpace& ex, IteratorType1 first, IteratorType1 last,
              IteratorType2 s_first, IteratorType2 s_last,
              const BinaryPredicateType& pred) {
  return Impl::find_first_of_exespace_impl(
      "Kokkos::find_first_of_iterator_api_default", ex, first, last, s_first,
      s_last, pred);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                  IteratorType1>
find_first_of(const std::string& label, const ExecutionSpace& ex,
              IteratorType1 first, IteratorType1 last, IteratorType2 s_first,
              IteratorType2 s_last, const BinaryPredicateType& pred) {
  return Impl::find_first_of_exespace_impl(label, ex, first, last, s_first,
                                           s_last, pred);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto find_first_of(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,
                   const ::Kokkos::View<DataType2, Properties2...>& s_view,
                   const BinaryPredicateType& pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_exespace_impl(
      "Kokkos::find_first_of_view_api_default", ex, KE::begin(view),
      KE::end(view), KE::begin(s_view), KE::end(s_view), pred);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType,
          std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value,
                            int> = 0>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,
                   const ::Kokkos::View<DataType2, Properties2...>& s_view,
                   const BinaryPredicateType& pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_exespace_impl(label, ex, KE::begin(view),
                                           KE::end(view), KE::begin(s_view),
                                           KE::end(s_view), pred);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

// overload set 1: no binary predicate passed
template <class TeamHandleType, class IteratorType1, class IteratorType2>
KOKKOS_FUNCTION
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, IteratorType1>
    find_first_of(const TeamHandleType& teamHandle, IteratorType1 first,
                  IteratorType1 last, IteratorType2 s_first,
                  IteratorType2 s_last) {
  return Impl::find_first_of_team_impl(teamHandle, first, last, s_first,
                                       s_last);
}

template <
    class TeamHandleType, class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto find_first_of(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& view,
    const ::Kokkos::View<DataType2, Properties2...>& s_view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_team_impl(teamHandle, KE::begin(view),
                                       KE::end(view), KE::begin(s_view),
                                       KE::end(s_view));
}

// overload set 2: binary predicate passed
template <class TeamHandleType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
KOKKOS_FUNCTION
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, IteratorType1>
    find_first_of(const TeamHandleType& teamHandle, IteratorType1 first,
                  IteratorType1 last, IteratorType2 s_first,
                  IteratorType2 s_last, const BinaryPredicateType& pred) {
  return Impl::find_first_of_team_impl(teamHandle, first, last, s_first, s_last,
                                       pred);
}

template <
    class TeamHandleType, class DataType1, class... Properties1,
    class DataType2, class... Properties2, class BinaryPredicateType,
    std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto find_first_of(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& view,
    const ::Kokkos::View<DataType2, Properties2...>& s_view,
    const BinaryPredicateType& pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(s_view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_first_of_team_impl(teamHandle, KE::begin(view),
                                       KE::end(view), KE::begin(s_view),
                                       KE::end(s_view), pred);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
