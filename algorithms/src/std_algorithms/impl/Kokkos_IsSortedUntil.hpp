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

#ifndef KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <std_algorithms/Kokkos_Find.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class ComparatorType, class ReducerType>
struct StdIsSortedUntilFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ComparatorType m_comparator;
  ReducerType m_reducer;

  KOKKOS_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto& val_i   = m_first[i];
    const auto& val_ip1 = m_first[i + 1];
    if (m_comparator(val_ip1, val_i)) {
      m_reducer.join(red_value, i);
    }
  }

  KOKKOS_FUNCTION
  StdIsSortedUntilFunctor(IteratorType first, ComparatorType comparator,
                          ReducerType reducer)
      : m_first(std::move(first)),
        m_comparator(std::move(comparator)),
        m_reducer(std::move(reducer)) {}
};

//
// overloads accepting exespace
//
template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until_exespace_impl(const std::string& label,
                                           const ExecutionSpace& ex,
                                           IteratorType first,
                                           IteratorType last,
                                           ComparatorType comp) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  const auto num_elements = Kokkos::Experimental::distance(first, last);

  // trivial case
  if (num_elements <= 1) {
    return last;
  }

  /*
    Do a par_reduce computing the *min* index that breaks the sorting.
    If one such index is found, then the range is sorted until that element,
    if no such index is found, then it means the range is sorted until the end.
  */
  using index_type = typename IteratorType::difference_type;
  index_type red_result;
  index_type red_result_init;
  ::Kokkos::Min<index_type> reducer(red_result);
  reducer.init(red_result_init);
  ::Kokkos::parallel_reduce(
      label,
      // use num_elements-1 because each index handles i and i+1
      RangePolicy<ExecutionSpace>(ex, 0, num_elements - 1),
      // use CTAD
      StdIsSortedUntilFunctor(first, comp, reducer), reducer);

  /* If the reduction result is equal to the initial value,
     and it means the range is sorted until the end */
  if (red_result == red_result_init) {
    return last;
  } else {
    /* If  such index is found, then the range is sorted until there and
       we need to return an iterator past the element found so do +1 */
    return first + (red_result + 1);
  }
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until_exespace_impl(const std::string& label,
                                           const ExecutionSpace& ex,
                                           IteratorType first,
                                           IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t     = Impl::StdAlgoLessThanBinaryPredicate<value_type>;
  return is_sorted_until_exespace_impl(label, ex, first, last, pred_t());
}

//
// overloads accepting team handle
//
template <class ExecutionSpace, class IteratorType, class ComparatorType>
KOKKOS_FUNCTION IteratorType
is_sorted_until_team_impl(const ExecutionSpace& teamHandle, IteratorType first,
                          IteratorType last, ComparatorType comp) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first);
  Impl::expect_valid_range(first, last);

  const auto num_elements = Kokkos::Experimental::distance(first, last);

  // trivial case
  if (num_elements <= 1) {
    return last;
  }

  /*
    Do a par_reduce computing the *min* index that breaks the sorting.
    If one such index is found, then the range is sorted until that element,
    if no such index is found, then it means the range is sorted until the end.
  */
  using index_type = typename IteratorType::difference_type;
  index_type red_result;
  index_type red_result_init;
  ::Kokkos::Min<index_type> reducer(red_result);
  reducer.init(red_result_init);
  ::Kokkos::parallel_reduce(  // use num_elements-1 because each index handles i
                              // and i+1
      TeamThreadRange(teamHandle, 0, num_elements - 1),
      // use CTAD
      StdIsSortedUntilFunctor(first, comp, reducer), reducer);
  teamHandle.team_barrier();

  /* If the reduction result is equal to the initial value,
     and it means the range is sorted until the end */
  if (red_result == red_result_init) {
    return last;
  } else {
    /* If  such index is found, then the range is sorted until there and
       we need to return an iterator past the element found so do +1 */
    return first + (red_result + 1);
  }
}

template <class ExecutionSpace, class IteratorType>
KOKKOS_FUNCTION IteratorType is_sorted_until_team_impl(
    const ExecutionSpace& teamHandle, IteratorType first, IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t     = Impl::StdAlgoLessThanBinaryPredicate<value_type>;
  return is_sorted_until_team_impl(teamHandle, first, last, pred_t());
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
