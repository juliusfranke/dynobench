#include "dynobench/integrator2_3d_coupled.hpp"
#include "Eigen/Core"
#include "dynobench/robot_models_base.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include <algorithm>
#include <cmath>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/ellipsoid.h>
#include <fcl/fcl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/node.h>
#include "dynobench/nn.h"

namespace dynobench {

void Integrator2_3d_coupled_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(distance_weights));
}

void Integrator2_3d_coupled_params::write(std::ostream &out) {

  const std::string be = "";
  const std::string af = ": ";

  out << be << STR(shape, af) << std::endl;
  out << be << STR(dt, af) << std::endl;
  out << be << STR(max_vel, af) << std::endl;
  out << be << STR(max_acc, af) << std::endl;
  out << be << STR(distance_weights, af) << std::endl;
  out << be << STR(filename, af) << std::endl;
}

void Integrator2_3d_coupled_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

Integrator2_3d_coupled::Integrator2_3d_coupled(const Integrator2_3d_coupled_params &params,
                               const Eigen::VectorXd &p_lb,
                               const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<Rn>(12), 6), params(params) {


  // description of state and control
  x_desc = {"x1[m]", "y1[m]", "z1[m]", "vx1[m/s]", "vy1[m/s]", "vz1[m/s]", "x2[m]", "y2[m]", "z2[m]", "vx2[m/s]", "vy2[m/s]", "vz2[m/s]"};
  u_desc = {"ax1[m/s^2]", "ay1[m/s^2]", "az1[m/s^2]","ax2[m/s^2]", "ay2[m/s^2]", "az2[m/s^2]"};

  is_2d = false;
  ts_data.resize(2);
  col_outs.resize(2);
  nx_col = 12;
  nx_pr = 12;
  translation_invariance = 3;

  distance_weights = params.distance_weights; // necessary for ompl wrapper
  name = "Integrator2_3d_coupled";

  ref_dt = params.dt;

  u_lb << params.min_acc, params.min_acc, params.min_acc, params.min_acc, params.min_acc, params.min_acc;
  u_ub << params.max_acc, params.max_acc, params.max_acc, params.max_acc, params.max_acc, params.max_acc;

  x_lb << low__, low__, low__, params.min_vel, params.min_vel, params.min_vel, low__, low__, low__, params.min_vel, params.min_vel, params.min_vel;
  x_ub << max__, max__, max__, params.max_vel, params.max_vel, params.max_vel, max__, max__, max__, params.max_vel, params.max_vel, params.max_vel;

  u_weight << 1., 1., 1., 1., 1., 1.;
  x_weightb << 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400;
  // add bounds on position if provided
  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }

  // collisions
  if (params.shape == "sphere") {
    collision_geometries.push_back(
        std::make_shared<fcl::Sphered>(params.radius));
    collision_geometries.push_back(
            std::make_shared<fcl::Sphered>(params.radius));
  } else if (params.shape == "ellipsoid") {
    collision_geometries.push_back(
        std::make_shared<fcl::Ellipsoidd>(params.radii));
    collision_geometries.push_back(
        std::make_shared<fcl::Ellipsoidd>(params.radii));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  part_objs_.clear();
  for (size_t i = 0; i < collision_geometries.size(); i++) {
    auto robot_part = new fcl::CollisionObject(collision_geometries[i]);
    part_objs_.push_back(robot_part);
  }

  col_mng_robots_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
  col_mng_robots_->setup();
}

Integrator2_3d_coupled::~Integrator2_3d_coupled()
{
  for (auto part : part_objs_) {
      delete part;
  }
}

double Integrator2_3d_coupled::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                 const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 4> maxs = {
      (x.head<3>() - y.head<3>()).norm() / params.max_vel,
      (x.segment<3>(3) - y.segment<3>(3)).norm() / params.max_acc,
      (x.segment<3>(6) - y.segment<3>(6)).norm() / params.max_vel,
      (x.tail<3>() - y.tail<3>()).norm() / params.max_acc};

  return *std::max_element(maxs.begin(), maxs.end());
}

void Integrator2_3d_coupled::set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) {
  x.segment<3>(3).setZero();
  x.tail<3>().setZero();
}

double Integrator2_3d_coupled::lower_bound_time_vel(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {
    std::array<double, 2> maxs = {
    (x.segment<3>(3) - y.segment<3>(3)).norm() / params.max_acc,
    (x.tail<3>() - y.tail<3>()).norm() / params.max_acc};

  return *std::max_element(maxs.begin(), maxs.end());
}

double Integrator2_3d_coupled::lower_bound_time_pr(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {
    std::array<double, 2> maxs = {
      (x.segment<3>(3) - y.segment<3>(3)).norm() / params.max_acc,
      (x.tail<3>() - y.tail<3>()).norm() / params.max_acc};

  return *std::max_element(maxs.begin(), maxs.end());
}

double Integrator2_3d_coupled::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &y) {

  assert(distance_weights.size() == 2);
  return params.distance_weights(0) * (x.head<3>() - y.head<3>()).norm() +
        params.distance_weights(1) * (x.segment<3>(3) - y.segment<3>(3)).norm() +
        params.distance_weights(0) * (x.segment<3>(6) - y.segment<3>(6)).norm() +
        params.distance_weights(1) * (x.tail<3>() - y.tail<3>()).norm();
};

void Integrator2_3d_coupled::calcV(Eigen::Ref<Eigen::VectorXd> v,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {

  Eigen::Vector3d a_repulsive;
  a_repulsive.setZero();
  // double alpha = 1;
  // auto p1 = x.head<3>();
  // auto p2 = x.segment<3>(6);
  // float dist = (p1-p2).norm();
  // a_repulsive = alpha / ( dist * dist) * (p2 - p1) / dist;
  auto dist = x.head<6>() - x.segment<6>(6);
  float input[6] = {static_cast<float>(dist(0)),
                    static_cast<float>(dist(1)),
                    static_cast<float>(dist(2)),
                    static_cast<float>(dist(3)),
                    static_cast<float>(dist(4)),
                    static_cast<float>(dist(5))};
  if(abs(dist(0)) < 0.2 && abs(dist(1)) < 0.2 && abs(dist(2)) < 1.5){ // only for x, y, z
    std::cout << "running nn to get fa" << std::endl;
    std::cout << "dist: " << dist.format(dynobench::FMT) << std::endl;
    nn_add_neighbor(input, NN_ROBOT_SMALL);
    const float* rhoOutput = nn_eval(NN_ROBOT_SMALL); // in gramms
    a_repulsive(2) = rhoOutput[0] / 1000 * 9.81; // in Newtons
    std::cout << "a_repulsive: " << a_repulsive.format(dynobench::FMT) << std::endl;
  }

  v(0) = x(3);
  v(1) = x(4);
  v(2) = x(5);
  v(3) = u(0) + a_repulsive(0);
  v(4) = u(1) + a_repulsive(1);
  v(5) = u(2) + a_repulsive(2);

  v(6) = x(9);
  v(7) = x(10);
  v(8) = x(11);
  v(9) = u(3) + a_repulsive(0);
  v(10) = u(4) + a_repulsive(1);
  v(11) = u(5) + a_repulsive(2);
}

// DYNAMICS
void Integrator2_3d_coupled::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                               Eigen::Ref<Eigen::MatrixXd> Jv_u,
                               const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u) {

  (void)x;
  (void)u;
  assert(x.size() == 12);
  assert(u.size() == 6);
  assert(Jv_x.rows() == 12);
  assert(Jv_x.cols() == 12);
  assert(Jv_u.rows() == 12);
  assert(Jv_u.cols() == 6);

  if(finite_diff){
    finite_diff_jac(
      [&](const Eigen::VectorXd &x_in, Eigen::Ref<Eigen::VectorXd> y) {
        calcV(y, x_in, u);
      },
      x, x.size(), Jv_x);

    finite_diff_jac(
      [&](const Eigen::VectorXd &u_in, Eigen::Ref<Eigen::VectorXd> y) {
        calcV(y, x, u_in);
      },
      u, x.size(), Jv_u);
  }
  else {
    Jv_x(0, 3) = 1;
    Jv_x(1, 4) = 1;
    Jv_x(2, 5) = 1;
    Jv_x(6, 9) = 1;
    Jv_x(7, 10) = 1;
    Jv_x(8, 11) = 1;

    Jv_u(3, 0) = 1;
    Jv_u(4, 1) = 1;
    Jv_u(5, 2) = 1;
    Jv_u(9, 3) = 1;
    Jv_u(10, 4) = 1;
    Jv_u(11, 5) = 1;

  }
}

// Collisions
void Integrator2_3d_coupled::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {

  assert(x.size() == 12);
  assert(ts.size() == 2);

  fcl::Transform3d result;
  fcl::Transform3d result2;

  result = Eigen::Translation<double, 3>(fcl::Vector3d(x(0), x(1), x(2)));
  result2 = Eigen::Translation<double, 3>(fcl::Vector3d(x(6), x(7), x(8)));

  ts.at(0) = result;
  ts.at(1) = result2;

}
void Integrator2_3d_coupled::collision_distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                     CollisionOut &cout) {
  double min_dist = std::numeric_limits<double>::max();
  bool check_parts = true;
  if (env) {
    transformation_collision_geometries(x, ts_data);
    DYNO_CHECK_EQ(collision_geometries.size(), ts_data.size(), AT);
    assert(collision_geometries.size() == ts_data.size());
    DYNO_CHECK_EQ(collision_geometries.size(), col_outs.size(), AT);
    assert(collision_geometries.size() == col_outs.size());
    robot_objs_.clear();
    col_mng_robots_->clear();
    for (size_t i = 0; i < ts_data.size(); i++) {
      fcl::Transform3d &transform = ts_data[i];
      auto robot_co = part_objs_[i];
      robot_co->setTranslation(transform.translation());
      robot_co->setRotation(transform.rotation());
      robot_co->computeAABB();
      robot_objs_.push_back(robot_co);
    }
    // part/environment checking
    for (size_t i = 0; i < ts_data.size(); i++) {
      auto robot_co = robot_objs_[i];
      fcl::DefaultDistanceData<double> distance_data;
      distance_data.request.enable_signed_distance = true;
      env->distance(robot_co, &distance_data,
                    fcl::DefaultDistanceFunction<double>);
      min_dist = std::min(min_dist, distance_data.result.min_distance);
    }

    if (check_parts) {
      col_mng_robots_->registerObjects(robot_objs_);
      fcl::DefaultDistanceData<double> inter_robot_distance_data;
      inter_robot_distance_data.request.enable_signed_distance = true;

      col_mng_robots_->distance(&inter_robot_distance_data,
                                fcl::DefaultDistanceFunction<double>);
      min_dist =
          std::min(min_dist, inter_robot_distance_data.result.min_distance);
    }
    cout.distance = min_dist;
  } else {
    cout.distance = max__;
  }
}
}; // namespace dynobench
