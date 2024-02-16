#pragma once

// Auto generated file
// Created at: 2023-09-06--10-43-58

namespace dynobench {

void calcV_n5_p(double *ff,

                double mp, double arm_length, double t2t, const double *m,
                const double *J_vx, const double *J_vy, const double *J_vz,
                const double *l, const double *x, const double *u);

void calcStep_n5_p(double *xnext, double mp, double arm_length, double t2t,
                   const double *m, const double *J_vx, const double *J_vy,
                   const double *J_vz, const double *l, const double *x,
                   const double *u, double dt);

void calcJ_n5_p(double *Jx, double *Ju,

                double mp, double arm_length, double t2t, const double *m,
                const double *J_vx, const double *J_vy, const double *J_vz,
                const double *l, const double *x, const double *u);

void calcF_n5_p(double *Fx, double *Fu,

                double mp, double arm_length, double t2t, const double *m,
                const double *J_vx, const double *J_vy, const double *J_vz,
                const double *l, const double *x, const double *u, double dt);

} // namespace dynobench
