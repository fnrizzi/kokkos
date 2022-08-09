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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON_HPP

#include <gtest/gtest.h>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Random.hpp>
#include <TestStdAlgorithmsHelperFunctors.hpp>
#include <utility>
#include <numeric>
#include <random>

namespace Test {
namespace stdalgos {

using exespace = Kokkos::DefaultExecutionSpace;

//
// tags
//
struct DynamicTag {};

// these are for rank-1
struct StridedTwoTag {};
struct StridedThreeTag {};

// these are for rank-2
struct StridedTwoRowsTag {};
struct StridedThreeRowsTag {};

const std::vector<int> teamSizesToTest = {1, 2, 23, 77, 123};

// map of scenarios where the key is a description
// and the value is the extent
const std::map<std::string, std::size_t> default_scenarios = {
    {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
    {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
    {"medium-a", 1003},    {"medium-b", 1003}, {"large-a", 101513},
    {"large-b", 101513}};

// see cpp file for these functions
std::string view_tag_to_string(DynamicTag);
std::string view_tag_to_string(StridedTwoTag);
std::string view_tag_to_string(StridedThreeTag);
std::string view_tag_to_string(StridedTwoRowsTag);
std::string view_tag_to_string(StridedThreeRowsTag);

//
// overload set for create_view for rank1
//

// dynamic
template <class ValueType>
auto create_view(DynamicTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*>;
  view_t view{label + "_" + view_tag_to_string(DynamicTag{}), ext};
  return view;
}

// stride2
template <class ValueType>
auto create_view(StridedTwoTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext, 2};
  view_t view{label + "_" + view_tag_to_string(StridedTwoTag{}), layout};
  return view;
}

// stride3
template <class ValueType>
auto create_view(StridedThreeTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext, 3};
  view_t view{label + "_" + view_tag_to_string(StridedThreeTag{}), layout};
  return view;
}

//
// overload set for create_view for rank2
//

// dynamic
template <class ValueType>
auto create_view(DynamicTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using view_t = Kokkos::View<ValueType**>;
  view_t view{label + "_" + view_tag_to_string(DynamicTag{}), ext0, ext1};
  return view;
}

// stride2rows
template <class ValueType>
auto create_view(StridedTwoRowsTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using view_t = Kokkos::View<ValueType**, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext0, 2, ext1, ext0 * 2};
  view_t view{label + "_" + view_tag_to_string(StridedTwoRowsTag{}), layout};
  return view;
}

// stride3rows
template <class ValueType>
auto create_view(StridedThreeRowsTag, std::size_t ext0, std::size_t ext1,
                 const std::string label) {
  using view_t = Kokkos::View<ValueType**, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext0, 3, ext1, ext0 * 3};
  view_t view{label + "_" + view_tag_to_string(StridedThreeRowsTag{}), layout};
  return view;
}

//
// overload set for create_deep_copyable_compatible_view_with_same_extent
//
// rank1
template <class ViewType, std::enable_if_t<ViewType::rank == 1, int> = 0>
auto create_deep_copyable_compatible_view_with_same_extent(ViewType view) {
  const std::size_t ext      = view.extent(0);
  using view_value_type      = typename ViewType::value_type;
  using view_exespace        = typename ViewType::execution_space;
  using view_deep_copyable_t = Kokkos::View<view_value_type*, view_exespace>;
  view_deep_copyable_t view_dc("view_dc", ext);
  return view_dc;
}

// rank2
template <class ViewType, std::enable_if_t<ViewType::rank == 2, int> = 0>
auto create_deep_copyable_compatible_view_with_same_extent(ViewType view) {
  const std::size_t ext0     = view.extent(0);
  const std::size_t ext1     = view.extent(1);
  using view_value_type      = typename ViewType::value_type;
  using view_exespace        = typename ViewType::execution_space;
  using view_deep_copyable_t = Kokkos::View<view_value_type**, view_exespace>;
  view_deep_copyable_t view_dc("view_dc", ext0, ext1);
  return view_dc;
}

//
// overload set for create_deep_copyable_compatible_clone
//
// rank1
template <class ViewType, std::enable_if_t<ViewType::rank == 1, int> = 0>
auto create_deep_copyable_compatible_clone(ViewType view) {
  auto view_dc    = create_deep_copyable_compatible_view_with_same_extent(view);
  using view_dc_t = decltype(view_dc);
  CopyFunctor<ViewType, view_dc_t> F1(view, view_dc);
  Kokkos::parallel_for("copy", view.extent(0), F1);
  return view_dc;
}

// rank2
template <class ViewType, std::enable_if_t<ViewType::rank == 2, int> = 0>
auto create_deep_copyable_compatible_clone(ViewType view) {
  auto view_dc    = create_deep_copyable_compatible_view_with_same_extent(view);
  using view_dc_t = decltype(view_dc);
  CopyFunctorRank2<ViewType, view_dc_t> F1(view, view_dc);
  Kokkos::parallel_for("copy", view.extent(0) * view.extent(1), F1);
  return view_dc;
}

//
// others
//

template <class LayoutTagType, class ValueType>
auto create_view_and_fill_randomly(LayoutTagType LayoutTag, std::size_t numRows,
                                   std::size_t numCols, ValueType lower,
                                   ValueType upper, const std::string& label) {
  // construct in memory space associated with default exespace
  auto dataView = create_view<ValueType>(LayoutTag, numRows, numCols, label);

  // dataView might not deep copyable (e.g. strided layout) so to
  // randomize it, we make a new view that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  // randomly fill the view
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(12371);
  Kokkos::fill_random(dataView_dc_h, pool, lower, upper);

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);
  // use CTAD
  CopyFunctorRank2 F1(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1), F1);

  return std::make_tuple(dataView, dataView_dc_h);
}

template <class ViewType>
auto create_host_space_copy(ViewType view) {
  auto view_dc = create_deep_copyable_compatible_clone(view);
  return create_mirror_view_and_copy(Kokkos::HostSpace(), view_dc);
}

// fill the views with sequentially increasing values
template <class ViewType, class ViewHostType>
void fill_views_inc(ViewType view, ViewHostType host_view) {
  namespace KE = Kokkos::Experimental;

  Kokkos::parallel_for(view.extent(0), AssignIndexFunctor<ViewType>(view));
  std::iota(KE::begin(host_view), KE::end(host_view), 0);
  // compare_views(expected, view);
}

template <class ValueType, class ViewType>
std::enable_if_t<!std::is_same<typename ViewType::traits::array_layout,
                               Kokkos::LayoutStride>::value>
verify_values(ValueType expected, const ViewType view) {
  static_assert(std::is_same<ValueType, typename ViewType::value_type>::value,
                "Non-matching value types of view and reference value");
  auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
  for (std::size_t i = 0; i < view_h.extent(0); i++) {
    EXPECT_EQ(expected, view_h(i));
  }
}

template <class ValueType, class ViewType>
std::enable_if_t<std::is_same<typename ViewType::traits::array_layout,
                              Kokkos::LayoutStride>::value>
verify_values(ValueType expected, const ViewType view) {
  static_assert(std::is_same<ValueType, typename ViewType::value_type>::value,
                "Non-matching value types of view and reference value");

  using non_strided_view_t = Kokkos::View<typename ViewType::value_type*>;
  non_strided_view_t tmpView("tmpView", view.extent(0));

  Kokkos::parallel_for(
      "_std_algo_copy", view.extent(0),
      CopyFunctor<ViewType, non_strided_view_t>(view, tmpView));
  auto view_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tmpView);
  for (std::size_t i = 0; i < view_h.extent(0); i++) {
    EXPECT_EQ(expected, view_h(i));
  }
}

template <class ViewType1, class ViewType2>
std::enable_if_t<!std::is_same<typename ViewType2::traits::array_layout,
                               Kokkos::LayoutStride>::value>
compare_views(ViewType1 expected, const ViewType2 actual) {
  static_assert(std::is_same<typename ViewType1::value_type,
                             typename ViewType2::value_type>::value,
                "Non-matching value types of expected and actual view");
  auto expected_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);
  auto actual_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);

  for (std::size_t i = 0; i < expected_h.extent(0); i++) {
    EXPECT_EQ(expected_h(i), actual_h(i));
  }
}

template <class ViewType1, class ViewType2>
std::enable_if_t<std::is_same<typename ViewType2::traits::array_layout,
                              Kokkos::LayoutStride>::value>
compare_views(ViewType1 expected, const ViewType2 actual) {
  static_assert(std::is_same<typename ViewType1::value_type,
                             typename ViewType2::value_type>::value,
                "Non-matching value types of expected and actual view");

  using non_strided_view_t = Kokkos::View<typename ViewType2::value_type*>;
  non_strided_view_t tmp_view("tmp_view", actual.extent(0));
  Kokkos::parallel_for(
      "_std_algo_copy", actual.extent(0),
      CopyFunctor<ViewType2, non_strided_view_t>(actual, tmp_view));

  auto actual_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tmp_view);
  auto expected_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);

  for (std::size_t i = 0; i < expected_h.extent(0); i++) {
    EXPECT_EQ(expected_h(i), actual_h(i));
  }
}

template <class ViewType1, class ViewType2>
std::enable_if_t<
    ViewType1::rank == 2 && ViewType2::rank == 2 &&
    std::is_same<typename ViewType1::memory_space, Kokkos::HostSpace>::value &&
    std::is_same<typename ViewType2::memory_space, Kokkos::HostSpace>::value>
expect_equal_host_views(ViewType1 A, const ViewType2 B) {
  EXPECT_EQ(A.extent(0), B.extent(0));
  EXPECT_EQ(A.extent(1), B.extent(1));

  for (std::size_t i = 0; i < A.extent(0); i++) {
    for (std::size_t j = 0; j < A.extent(1); j++) {
      EXPECT_EQ(A(i, j), B(i, j));
    }
  }
}

template <class ViewType>
void fill_zero(ViewType a) {
  const auto functor = FillZeroFunctor<ViewType>(a);
  ::Kokkos::parallel_for(a.extent(0), std::move(functor));
}

template <class ViewType1, class ViewType2>
void fill_zero(ViewType1 a, ViewType2 b) {
  fill_zero(a);
  fill_zero(b);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// helpers for testing small views (extent = 10)
// prefer `default_scenarios` map for creating new tests
using value_type = double;

struct std_algorithms_test : public ::testing::Test {
  static constexpr size_t extent = 10;

  using static_view_t = Kokkos::View<value_type[extent]>;
  static_view_t m_static_view{"std-algo-test-1D-contiguous-view-static"};

  using dyn_view_t = Kokkos::View<value_type*>;
  dyn_view_t m_dynamic_view{"std-algo-test-1D-contiguous-view-dynamic", extent};

  using strided_view_t = Kokkos::View<value_type*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{extent, 2};
  strided_view_t m_strided_view{"std-algo-test-1D-strided-view", layout};

  using view_host_space_t = Kokkos::View<value_type[10], Kokkos::HostSpace>;

  template <class ViewFromType>
  void copyInputViewToFixtureViews(ViewFromType view) {
    CopyFunctor<ViewFromType, static_view_t> F1(view, m_static_view);
    Kokkos::parallel_for("_std_algo_copy1", view.extent(0), F1);

    CopyFunctor<ViewFromType, dyn_view_t> F2(view, m_dynamic_view);
    Kokkos::parallel_for("_std_algo_copy2", view.extent(0), F2);

    CopyFunctor<ViewFromType, strided_view_t> F3(view, m_strided_view);
    Kokkos::parallel_for("_std_algo_copy3", view.extent(0), F3);
  }
};

struct CustomValueType {
  KOKKOS_INLINE_FUNCTION
  CustomValueType(){};

  KOKKOS_INLINE_FUNCTION
  CustomValueType(value_type val) : value(val){};

  KOKKOS_INLINE_FUNCTION
  CustomValueType(const CustomValueType& other) { this->value = other.value; }

  KOKKOS_INLINE_FUNCTION
  explicit operator value_type() const { return value; }

  KOKKOS_INLINE_FUNCTION
  value_type& operator()() { return value; }

  KOKKOS_INLINE_FUNCTION
  const value_type& operator()() const { return value; }

  KOKKOS_INLINE_FUNCTION
  CustomValueType& operator+=(const CustomValueType& other) {
    this->value += other.value;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CustomValueType& operator=(const CustomValueType& other) {
    this->value = other.value;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CustomValueType operator+(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value + other.value;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  CustomValueType operator-(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value - other.value;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  CustomValueType operator*(const CustomValueType& other) const {
    CustomValueType result;
    result.value = this->value * other.value;
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const CustomValueType& other) const {
    return this->value == other.value;
  }

 private:
  friend std::ostream& operator<<(std::ostream& os,
                                  const CustomValueType& custom_value_type) {
    return os << custom_value_type.value;
  }
  value_type value = {};
};

}  // namespace stdalgos
}  // namespace Test

#endif
