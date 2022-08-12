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

namespace Test {
namespace stdalgos {
namespace TeamFill_n {

namespace KE = Kokkos::Experimental;

template <class ViewType, class DistancesViewType>
struct TestFunctorA {
  ViewType m_view;
  DistancesViewType m_distancesView;
  std::size_t m_fillCount;
  int m_apiPick;

  TestFunctorA(const ViewType view, const DistancesViewType distancesView,
               std::size_t fillCount, int apiPick)
      : m_view(view),
        m_distancesView(distancesView),
        m_fillCount(fillCount),
        m_apiPick(apiPick) {}

  template <class MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& member) const {
    const auto leagueRank = member.league_rank();
    const auto myRowIndex = leagueRank;
    auto myRowView        = Kokkos::subview(m_view, myRowIndex, Kokkos::ALL());

    if (m_apiPick == 0) {
      auto it =
          KE::fill_n(member, KE::begin(myRowView), m_fillCount, leagueRank);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    } else if (m_apiPick == 1) {
      auto it = KE::fill_n(member, myRowView, m_fillCount, leagueRank);

      Kokkos::single(Kokkos::PerTeam(member), [=]() {
        m_distancesView(myRowIndex) = KE::distance(KE::begin(myRowView), it);
      });
    }
  }
};

template <class LayoutTag, class ValueType>
void test_A(std::size_t numTeams, std::size_t numCols, std::size_t fillCount,
            int apiId) {
  /* description:
     create a rank-2 view, run a team parfor with one row per team,
     such that n elements of each row are filled up with the league_rank value
     of the team in charge of it, while the other elements in the row
     are left unchanged
   */

  // -----------------------------------------------
  // prepare data
  // -----------------------------------------------
  // create a view in the memory space associated with default exespace
  // with as many rows as the number of teams and fill it with random
  // values from an arbitrary range
  auto [dataView, dataViewBeforeOp_h] = create_view_and_fill_randomly(
      LayoutTag{}, numTeams, numCols,
      Kokkos::pair{ValueType(5), ValueType(523)}, "dataView");

  // -----------------------------------------------
  // launch kokkos kernel
  // -----------------------------------------------
  using space_t = Kokkos::DefaultExecutionSpace;
  Kokkos::TeamPolicy<space_t> policy(numTeams, Kokkos::AUTO());

  // fill_n returns an iterator so to verify that it is correct
  // each team stores the distance of the returned iterator from the
  // beginning of the interval that team operates on and then we check
  // that these distances match the expected value
  Kokkos::View<std::size_t*> distancesView("distancesView", numTeams);

  // use CTAD for functor
  TestFunctorA fnc(dataView, distancesView, fillCount, apiId);
  Kokkos::parallel_for(policy, fnc);

  // -----------------------------------------------
  // check
  // -----------------------------------------------
  auto dataViewAfterOp_h = create_host_space_copy(dataView);
  auto distancesView_h   = create_host_space_copy(distancesView);
  for (std::size_t i = 0; i < dataView.extent(0); ++i) {
    // check that values match what we expect
    for (std::size_t j = 0; j < fillCount; ++j) {
      EXPECT_EQ(dataViewAfterOp_h(i, j), ValueType(i));
    }
    // all other elements should be unchanged from before op
    for (std::size_t j = fillCount; j < numCols; ++j) {
      EXPECT_EQ(dataViewAfterOp_h(i, j), dataViewBeforeOp_h(i, j));
    }

    // check that returned iterators are correct
    if (fillCount > 0) {
      EXPECT_EQ(distancesView_h(i), std::size_t(fillCount));
    } else {
      EXPECT_EQ(distancesView_h(i), std::size_t(0));
    }
  }
}

template <class Tag, class ValueType>
void run_all_scenarios() {
  // prepare a map where, for a given set of num cols
  // we provide a list of counts of elements to fill
  // key = num of columns,
  // value = list of num of elemenents to fill
  const std::map<int, std::vector<int>> scenarios = {
      {0, {0}},
      {2, {0, 1, 2}},
      {6, {0, 1, 2, 5}},
      {13, {0, 1, 2, 8, 11}},
      {56, {0, 1, 2, 8, 11, 33, 56}},
      {123, {0, 1, 11, 33, 56, 89, 112}}};

  for (int num_teams : teamSizesToTest) {
    for (const auto& scenario : scenarios) {
      const std::size_t numCols = scenario.first;
      for (int numFills : scenario.second) {
        for (int apiId : {0, 1}) {
          test_A<Tag, ValueType>(num_teams, numCols, numFills, apiId);
        }
      }
    }
  }
}

TEST(std_algorithms_fill_n_team_test, test) {
  run_all_scenarios<DynamicTag, double>();
  run_all_scenarios<StridedTwoRowsTag, int>();
  run_all_scenarios<StridedThreeRowsTag, unsigned>();
}

}  // namespace TeamFill_n
}  // namespace stdalgos
}  // namespace Test
