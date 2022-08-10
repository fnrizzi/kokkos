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

#ifndef KOKKOS_STD_ALGORITHMS_GENERATE_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_GENERATE_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class Generator>
struct StdGenerateFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  Generator m_generator;

  KOKKOS_FUNCTION
  void operator()(index_type i) const { m_first[i] = m_generator(); }

  KOKKOS_FUNCTION
  StdGenerateFunctor(IteratorType _first, Generator _g)
      : m_first(std::move(_first)), m_generator(std::move(_g)) {}
};

template <class ExecutionSpace, class IteratorType, class Generator>
void generate_exespace_impl(const std::string& label, const ExecutionSpace& ex,
                            IteratorType first, IteratorType last,
                            Generator g) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  // run
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  ::Kokkos::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         // use CTAD
                         StdGenerateFunctor(first, g));
  ex.fence("Kokkos::generate: fence after operation");
}

template <class ExecutionSpace, class IteratorType, class Size, class Generator>
IteratorType generate_n_impl(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, Size count, Generator g) {
  if (count <= 0) {
    return first;
  }

  generate_exespace_impl(label, ex, first, first + count, g);
  return first + count;
}

//
// team-level impl
//
template <class TeamHandleType, class IteratorType, class Generator>
KOKKOS_FUNCTION void generate_team_impl(const TeamHandleType& teamHandle,
                                        IteratorType first, IteratorType last,
                                        Generator g) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first);
  Impl::expect_valid_range(first, last);

  // run
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  ::Kokkos::parallel_for(TeamThreadRange(teamHandle, 0, num_elements),
                         StdGenerateFunctor(first, g));
  teamHandle.team_barrier();
}

template <class TeamHandleType, class IteratorType, class Size, class Generator>
KOKKOS_FUNCTION IteratorType
generate_n_team_impl(const TeamHandleType& teamHandle, IteratorType first,
                     Size count, Generator g) {
  if (count <= 0) {
    return first;
  }

  generate_team_impl(teamHandle, first, first + count, g);
  return first + count;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
