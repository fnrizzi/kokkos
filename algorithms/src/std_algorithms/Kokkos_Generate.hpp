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

#ifndef KOKKOS_STD_ALGORITHMS_GENERATE_HPP
#define KOKKOS_STD_ALGORITHMS_GENERATE_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_GenerateGenerateN.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class IteratorType, class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value> generate(
    const ExecutionSpace& ex, IteratorType first, IteratorType last,
    Generator g) {
  Impl::generate_exespace_impl("Kokkos::generate_iterator_api_default", ex,
                               first, last, std::move(g));
}

template <class ExecutionSpace, class IteratorType, class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value> generate(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, Generator g) {
  Impl::generate_exespace_impl(label, ex, first, last, std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value> generate(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType, Properties...>& view, Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  Impl::generate_exespace_impl("Kokkos::generate_view_api_default", ex,
                               begin(view), end(view), std::move(g));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Generator>
std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value> generate(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType, Properties...>& view, Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  Impl::generate_exespace_impl(label, ex, begin(view), end(view), std::move(g));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class IteratorType, class Generator>
KOKKOS_FUNCTION std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value>
generate(const TeamHandleType& teamHandle, IteratorType first,
         IteratorType last, Generator g) {
  Impl::generate_team_impl(teamHandle, first, last, std::move(g));
}

template <class TeamHandleType, class DataType, class... Properties,
          class Generator>
KOKKOS_FUNCTION std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value>
generate(const TeamHandleType& teamHandle,
         const ::Kokkos::View<DataType, Properties...>& view, Generator g) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::generate_team_impl(teamHandle, begin(view), end(view), std::move(g));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
