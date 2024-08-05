#include "dynobench/motions.hpp"
#include <vector>
#include <bits/stdc++.h>
#include <dynobench/multirobot_trajectory.hpp>

void from_joint_to_indiv_trajectory_meta(
    const std::unordered_set<size_t> &cluster,
    const dynobench::Trajectory &traj, 
    MultiRobotTrajectory &init_guess_multi_robot,
    MultiRobotTrajectory &solution_multi_robot,
    const std::vector<int> &times) {

  MultiRobotTrajectory multi_robot_traj;

  std::vector<int> nxs = init_guess_multi_robot.get_nxs();
  std::vector<int> nus = init_guess_multi_robot.get_nus();

  DYNO_CHECK_EQ(nxs.size(), nus.size(), "");
  DYNO_CHECK_EQ(nxs.size(), times.size(), "");

  size_t num_robots = nxs.size();

  std::vector<size_t> nxs_accumulated(cluster.size());
  std::vector<size_t> nus_accumulated(cluster.size());
  size_t j = 0;
  size_t id = 0; // keep track of the last robot in cluster
  for (size_t i = 0; i < num_robots; i++) { // only for cluster
    if(cluster.find(i) != cluster.end()){
      if(j == 0){
        nxs_accumulated.at(j) = 0;
        nus_accumulated.at(j) = 0;
        id = i;
        ++j;
      }
      else{
        nxs_accumulated.at(j) = nxs_accumulated.at(j - 1) + nxs.at(id);
        nus_accumulated.at(j) = nus_accumulated.at(j - 1) + nus.at(id);
        id = i;
        ++j;
      }
    }
  }

  j = 0;
  for (size_t i = 0; i < num_robots; i++) {
    if(cluster.find(i) != cluster.end()){ // only for those inside cluster
      dynobench::Trajectory traj_out;
      for (int k = 0; k < times.at(i); k++) {
        traj_out.states.push_back(
            traj.states.at(k).segment(nxs_accumulated.at(j), nxs.at(i)));
        if (k < times.at(i)-1)
          traj_out.actions.push_back(
              traj.actions.at(k).segment(nus_accumulated.at(j), nus.at(i)));
      } 
      solution_multi_robot.trajectories.at(i).states.resize(traj_out.states.size());
      solution_multi_robot.trajectories.at(i).actions.resize(traj_out.actions.size());
      solution_multi_robot.trajectories.at(i) = traj_out; 
      ++j; // keep track for cluster
    }
  }
}