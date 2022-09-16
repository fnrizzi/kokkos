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

#include <TestStdAlgorithmsCommon.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Test {
namespace stdalgos {
namespace TeamSearchN {

namespace KE = Kokkos::Experimental;

template <class ValueType>
struct EqualFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType& lhs, const ValueType& rhs) const {
    return lhs == rhs;
  }
};

template <class DataViewType, class SearchedValuesViewType,
          class DistancesViewType, class BinaryOpType>
struct TestFunctorA {
  DataViewType m_dataView;
  std::size_t m_seqSize;
  SearchedValuesViewType m_searchedValuesView;
  DistancesViewType m_distancesView;
  BinaryOpType m_binaryOp;
  int m_apiPick;

  TestFunctorA(const DataViewType dataView, std::size_t seqSize,
               const SearchedValuesViewType searchedValuesView,
               const DistancesViewType distancesView, BinaryOpType binaryOp,
               int apiPick)
      : m_dataView(dataView),
        m_seqSize(seqSize),
        m_searchedValuesView(searchedValuesView),
        m_distancesView(distancesView),
        m_binaryOp(std::move(binaryOp)),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto myRowIndex = member.league_rank();
    auto myRowViewFrom = Kokkos::subview(m_dataView, myRowIndex, Kokkos::ALL());
    auto rowFromBegin  = KE::begin(myRowViewFrom);
    auto rowFromEnd    = KE::end(myRowViewFrom);
    const auto searchedValue = m_searchedValuesView(myRowIndex);

    switch (m_apiPick) {
      case 0: {
        const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                     m_seqSize, searchedValue);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 1: {
        const auto it =
            KE::search_n(member, myRowViewFrom, m_seqSize, searchedValue);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 2: {
        const auto it = KE::search_n(member, rowFromBegin, rowFromEnd,
                                     m_seqSize, searchedValue, m_binaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }

      case 3: {
        const auto it = KE::search_n(member, myRowViewFrom, m_seqSize,
                                     searchedValue, m_binaryOp);
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
          m_distancesView(myRowIndex) = KE::distance(rowFromBegin, it);
        });

        break;
      }
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(const bool sequencesExist, std::size_t numTeams,
            std::size_t numCols, int apiId) {
  /* description:
     use a rank-2 view randomly filled with values,
     and run a team-level search_n
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range.
  const ValueType lowerBound{5}, upperBound{523};
  auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
      LayoutTag{}, numTeams, numCols, Kokkos::pair{lowerBound, upperBound},
      "dataView");

  // If sequencesExist == true we need to inject some sequence of count test
  // value into dataView. If sequencesExist == false we set searchedVal to a
  // value that is not present in dataView

  const std::size_t halfCols = (numCols > 1) ? ((numCols + 1) / 2) : (1);
  const std::size_t seqSize  = (numCols > 1) ? (std::log2(numCols)) : (1);

  Kokkos::View<ValueType*> searchedValuesView("searchedValuesView", numTeams);
  auto searchedValuesView_h = create_host_space_copy(searchedValuesView);

  // dataView might not deep copyable (e.g. strided layout) so to prepare it
  // correclty, we make a new view that is for sure deep copyable, modify it
  // on the host, deep copy to device and then launch a kernel to copy to
  // dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  if (sequencesExist) {
    const std::size_t dataBegin = halfCols - seqSize;
    for (std::size_t i = 0; i < searchedValuesView.extent(0); ++i) {
      const ValueType searchedVal = dataView_dc_h(i, dataBegin);
      searchedValuesView_h(i)     = searchedVal;

      for (std::size_t j = dataBegin + 1; j < seqSize; ++j) {
        dataView_dc_h(i, j) = searchedVal;
      }
    }

    // copy to dataView_dc and then to dataView
    Kokkos::deep_copy(dataView_dc, dataView_dc_h);

    CopyFunctorRank2 cpFun(dataView_dc, dataView);
    Kokkos::parallel_for("copy", dataView.extent(0) * dataView.extent(1),
                         cpFun);
  } else {
    using rand_pool =
        Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
    rand_pool pool(lowerBound * upperBound);
    Kokkos::fill_random(searchedValuesView_h, pool, upperBound, upperBound * 2);
  }

  Kokkos::deep_copy(searchedValuesView, searchedValuesView_h);

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // search_n returns an iterator so to verify that it is correct each team
  // stores the distance of the returned iterator from the beginning of the
  // interval that team operates on and then we check that these distances match
  // the std result
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  EqualFunctor<ValueType> binaryOp;

  // use CTAD for functor
  TestFunctorA fnc(dataView, seqSize, searchedValuesView, distancesView,
                   binaryOp, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // run cpp-std kernel and check
  // -----------------------------------------------
  auto distancesView_h = create_host_space_copy(distancesView);

  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    auto rowFrom = Kokkos::subview(dataView_dc_h, i, Kokkos::ALL());

    const auto rowFromBegin = KE::cbegin(rowFrom);
    const auto rowFromEnd   = KE::cend(rowFrom);

    const ValueType searchedVal = searchedValuesView_h(i);

    const std::size_t beginEndDist = KE::distance(rowFromBegin, rowFromEnd);

    switch (apiId) {
      case 0:
      case 1: {
        const auto it =
            std::search_n(rowFromBegin, rowFromEnd, seqSize, searchedVal);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          EXPECT_LT(distancesView_h(i), beginEndDist);
        } else {
          EXPECT_EQ(distancesView_h(i), beginEndDist);
        }

        EXPECT_EQ(stdDistance, distancesView_h(i));

        break;
      }

      case 2:
      case 3: {
        const auto it = std::search_n(rowFromBegin, rowFromEnd, seqSize,
                                      searchedVal, binaryOp);
        const std::size_t stdDistance = KE::distance(rowFromBegin, it);

        if (sequencesExist) {
          EXPECT_LT(distancesView_h(i), beginEndDist);
        } else {
          EXPECT_EQ(distancesView_h(i), beginEndDist);
        }

        EXPECT_EQ(stdDistance, distancesView_h(i));

        break;
      }
    }
  }
}

template <class LayoutTag, class ValueType>
void run_all_scenarios(const bool sequencesExist) {
  for (int numTeams : teamSizesToTest) {
    for (const auto& numCols : {2, 13, 101, 1444, 8153}) {
      for (int apiId : {0, 1}) {
        test_A<LayoutTag, ValueType>(sequencesExist, numTeams, numCols, apiId);
      }
    }
  }
}

TEST(std_algorithms_search_n_team_test, sequences_of_equal_elements_exist) {
  constexpr bool sequencesExist = true;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

TEST(std_algorithms_search_n_team_test,
     sequences_of_equal_elements_probably_does_not_exist) {
  constexpr bool sequencesExist = false;

  run_all_scenarios<DynamicTag, double>(sequencesExist);
  run_all_scenarios<StridedTwoRowsTag, int>(sequencesExist);
  run_all_scenarios<StridedThreeRowsTag, unsigned>(sequencesExist);
}

}  // namespace TeamSearchN
}  // namespace stdalgos
}  // namespace Test
