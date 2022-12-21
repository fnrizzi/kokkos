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

#ifndef KOKKOS_STD_ALGORITHMS_REPLACE_IF_HPP
#define KOKKOS_STD_ALGORITHMS_REPLACE_IF_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_ReplaceIf.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class InputIterator, class Predicate,
          class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value>
    replace_if(const ExecutionSpace& ex, InputIterator first,
               InputIterator last, Predicate pred, const ValueType& new_value) {
  return Impl::replace_if_impl(ex, first, last, pred, new_value,
                               "Kokkos::replace_if_iterator_api");
}

template <class ExecutionSpace, class InputIterator, class Predicate,
          class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value>
    replace_if(const std::string& label, const ExecutionSpace& ex,
               InputIterator first, InputIterator last, Predicate pred,
               const ValueType& new_value) {
  return Impl::replace_if_impl(ex, first, last, pred, new_value, label);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class Predicate, class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value>
    replace_if(const ExecutionSpace& ex,
               const ::Kokkos::View<DataType1, Properties1...>& view,
               Predicate pred, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_if_impl(ex, KE::begin(view), KE::end(view), pred,
                               new_value, "Kokkos::replace_if_view_api");
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class Predicate, class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value>
    replace_if(const std::string& label, const ExecutionSpace& ex,
               const ::Kokkos::View<DataType1, Properties1...>& view,
               Predicate pred, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_if_impl(ex, KE::begin(view), KE::end(view), pred,
                               new_value, label);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class InputIterator, class Predicate,
          class ValueType>
KOKKOS_FUNCTION std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value>
replace_if(const TeamHandleType& teamHandle, InputIterator first,
           InputIterator last, Predicate pred, const ValueType& new_value) {
  return Impl::replace_if_impl(teamHandle, first, last, pred, new_value);
}

template <class TeamHandleType, class DataType1, class... Properties1,
          class Predicate, class ValueType>
KOKKOS_FUNCTION std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value>
replace_if(const TeamHandleType& teamHandle,
           const ::Kokkos::View<DataType1, Properties1...>& view,
           Predicate pred, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_if_impl(teamHandle, KE::begin(view), KE::end(view), pred,
                               new_value);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
