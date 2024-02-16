
#include "dynobench/dyno_macros.hpp"
#include "dynobench/for_each_macro.hpp"
#include "dynobench/robot_models_base.hpp"

namespace dynobench {

struct Quad3dpayload_params {

  Quad3dpayload_params(const char *file) { read_from_yaml(file); }
  Quad3dpayload_params() = default;

  double col_size_robot = .2;    // radius
  double col_size_payload = .05; // radius

  double max_vel = 4;
  double max_angular_vel = 8;

  double max_acc = 25;
  double max_angular_acc = 20;

  bool motor_control = true;

  double m = 0.034; // kg

  double m_payload = 0.0054; // kg
  double l_payload = 0.5;    // m  Khaled DONE

  double g = 9.81;
  double max_f = 1.3;        // thrust to weight ratio -- Khaled DONE
  double arm_length = 0.046; // m
  double t2t = 0.006;        // thrust-to-torque ratio
  double dt = .01;
  std::string shape = "sphere";
  Eigen::Vector4d distance_weights = Eigen::Vector4d(1, 1, .1, .1);
  Eigen::Vector4d u_ub;
  Eigen::Vector4d u_lb;

  Eigen::Vector3d J_v =
      Eigen::Vector3d(16.571710e-6, 16.655602e-6, 29.261652e-6);

  Eigen::VectorXd size = Eigen::Matrix<double, 1, 1>(.4);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void write(std::ostream &out) {
    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(col_size_robot, af) << std::endl;
    out << be << STR(col_size_payload, af) << std::endl;

    out << be << STR(m_payload, af) << std::endl;
    out << be << STR(l_payload, af) << std::endl;

    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(max_acc, af) << std::endl;
    out << be << STR(max_angular_acc, af) << std::endl;
    out << be << STR(motor_control, af) << std::endl;
    out << be << STR(m, af) << std::endl;
    out << be << STR(g, af) << std::endl;
    out << be << STR(max_f, af) << std::endl;
    out << be << STR(arm_length, af) << std::endl;
    out << be << STR(t2t, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(filename, af) << std::endl;

    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR_VV(J_v, af) << std::endl;
    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(u_lb, af) << std::endl;
    out << be << STR_VV(u_ub, af) << std::endl;
  }
};

struct Model_quad3dpayload : Model_robot {

  using Vector19d = Eigen::Matrix<double, 19, 1>; // TODO: Khaled DONE
  using Matrix34 = Eigen::Matrix<double, 3, 4>;

  virtual ~Model_quad3dpayload() = default;

  struct Data {
    Eigen::Vector3d f_u;
    Eigen::Vector3d tau_u;
    Eigen::Matrix<double, 19, 1> xnext;
    Matrix34 Jx;
    Eigen::Matrix3d Ja;
  } data;

  Vector19d ff; // TODO: Khaled CHECK and adapt
  Quad3dpayload_params params;

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    // state (size): [x_load(3,)  q_cable(3,)   v_load(3,)   w_cable(3,)
    // quat(4,)     w_uav(3)]
    //         idx:  [(0, 1, 2), (3,  4,  5),  (6,  7,  8), (9,  10, 11),
    //         (12,13,14,15), (16, 17, 18)]
    // x.segment<6>(7).setZero();
    x.segment<6>(6).setZero();
    x.segment<3>(16).setZero();
    // NOT_IMPLEMENTED;
  }

  using State_components =
      std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d,
                 Eigen::Vector3d, Eigen::Vector4d, Eigen::Vector3d>;

  void get_state_components(const Eigen::Ref<const Eigen::VectorXd> &x,
                            State_components &out) {
    get_payload_pos(x, std::get<0>(out));
    get_qc(x, std::get<1>(out));
    get_vel(x, std::get<2>(out));
    get_wc(x, std::get<3>(out));
    get_q(x, std::get<4>(out));
    get_w(x, std::get<5>(out));
  }

  void get_payload_pos(const Eigen::Ref<const Eigen::VectorXd> &x,
                       Eigen::Ref<Eigen::Vector3d> out) {
    out = x.head<3>();
  }

  void get_qc(const Eigen::Ref<const Eigen::VectorXd> &x,
              Eigen::Ref<Eigen::Vector3d> out) {
    out = x.segment<3>(3);
  }

  void get_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
               Eigen::Ref<Eigen::Vector3d> out) {
    out = x.segment<3>(6);
  }

  void get_wc(const Eigen::Ref<const Eigen::VectorXd> &x,
              Eigen::Ref<Eigen::Vector3d> out) {
    out = x.segment<3>(9);
  }

  void get_q(const Eigen::Ref<const Eigen::VectorXd> &x,
             Eigen::Ref<Eigen::Vector4d> out) {
    out = x.segment<4>(12);
  }

  void get_w(const Eigen::Ref<const Eigen::VectorXd> &x,
             Eigen::Ref<Eigen::Vector3d> out) {
    out = x.segment<3>(16);
  }

  virtual void get_position_robot(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  Eigen::Ref<Eigen::Vector3d> out) {
    Eigen::Vector3d pp, qc;
    get_payload_pos(x, pp);
    get_qc(x, qc);
    out = pp - params.l_payload * qc;
  }

  virtual void
  get_position_center_cable(const Eigen::Ref<const Eigen::VectorXd> &x,
                            Eigen::Ref<Eigen::Vector3d> out) {
    Eigen::Vector3d pp, qc;
    get_payload_pos(x, pp);
    get_qc(x, qc);
    out = pp - .5 * params.l_payload * qc;
  }

  // NOTE: there are infinite solutions to this problem
  // we just take the "smallest" rotation
  // I just way this to update the capsule orientation
  virtual void quaternion_cable_(const Eigen::Ref<const Eigen::VectorXd> &x,
                                 Eigen::Ref<Eigen::Vector4d> out) {

    Eigen::Vector3d from(0., 0., -1.);
    Eigen::Vector3d to;
    get_qc(x, to);
    out = Eigen::Quaternion<double>::FromTwoVectors(from, to).coeffs();
  }

  double arm;
  double g = 9.81;

  double u_nominal;
  double m_inv;
  double m;
  Eigen::Vector3d inverseJ_v;

  Eigen::Matrix3d inverseJ_M;
  Eigen::Matrix3d J_M;

  Eigen::Matrix3d inverseJ_skew;
  Eigen::Matrix3d J_skew;

  Eigen::Vector3d grav_v;

  Eigen::Matrix4d B0;
  Eigen::Matrix4d B0inv;

  Matrix34 Fu_selection;
  Matrix34 Ftau_selection;

  Matrix34 Fu_selection_B0;
  Matrix34 Ftau_selection_B0;

  const bool adapt_vel = true;

  Model_quad3dpayload(const Model_quad3dpayload &) = default;

  Model_quad3dpayload(const char *file,
                      const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                      const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_quad3dpayload(Quad3dpayload_params(file), p_lb, p_ub) {}

  Model_quad3dpayload(
      const Quad3dpayload_params &params = Quad3dpayload_params(),
      const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
      const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void ensure(Eigen::Ref<Eigen::VectorXd> xout) override {
    // state (size): [x_load(3,)  q_cable(3,)   v_load(3,)   w_cable(3,)
    // quat(4,)     w_uav(3)]
    //         idx:  [(0, 1, 2), (3,  4,  5),  (6,  7,  8), (9,  10, 11),
    //         (12,13,14,15), (16, 17, 18)]
    xout.segment<4>(12).normalize();
    xout.segment<3>(3).normalize();
  }

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) override;

  virtual void
  motorForcesFromThrust(Eigen::Ref<Eigen::VectorXd> f,
                        const Eigen::Ref<const Eigen::VectorXd> tm) {

    // Eigen::Vector4d eta = B0 * u_nominal * f;
    // f_u << 0, 0, eta(0);
    // tau_u << eta(1), eta(2), eta(3);
    f = B0inv * tm / u_nominal;
  }

  virtual void transform_primitive(
      const Eigen::Ref<const Eigen::VectorXd> &p,
      const std::vector<Eigen::VectorXd> &xs_in,
      const std::vector<Eigen::VectorXd> &us_in, TrajWrapper &traj_out,
      // std::vector<Eigen::VectorXd> &xs_out,
      // std::vector<Eigen::VectorXd> &us_out,
      std::function<bool(Eigen::Ref<Eigen::VectorXd>)> *is_valid_fun = nullptr,
      int *num_valid_states = nullptr) override {
    NOT_IMPLEMENTED;
  }

  virtual void offset(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> p) override {
    // Not sure what to do here
    NOT_IMPLEMENTED;
  }

  virtual size_t get_offset_dim() override {
    // Not sure what to do here
    NOT_IMPLEMENTED;
  }

  virtual void canonical_state(const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {
    // Not sure what to do here
    NOT_IMPLEMENTED;
  }

  virtual void transform_state(const Eigen::Ref<const Eigen::VectorXd> &p,
                               const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {
    // Not sure what to do here
    NOT_IMPLEMENTED;
  }

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u,
                    double dt) override;

  virtual void stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u,
                        double dt) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x,
      std::vector<Transform3d> &ts) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void collision_distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  CollisionOut &cout) override;

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override;
};

} // namespace dynobench
