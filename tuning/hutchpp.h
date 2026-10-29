// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __HUTCHINSONPP_H__
#define __HUTCHINSONPP_H__

#include <random>
#include <Eigen/Dense>
#include <optional>
#include "linear_operator.h"
#include "./fdaPDE/src/models/header_check.h"


namespace fdapde {
namespace internals {

// Stochastic trace estimation by Hutch++ method, as described in:
// Raphael A. Meyer, Cameron Musco, Christopher Musco, David P. Woodruff (2020) Hutch++: Optimal Stochastic Trace Estimation

template <LinearOperator Op>

class Hutchpp {
    
    public:

    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    Hutchpp() noexcept = default;
    Hutchpp(const Op& A, int r = 100, int seed = random_seed) : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { compute(A); }
    Hutchpp(int r, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { }

    
    void compute(const Op& A) const {
        const int n = A.rows();
        // Defining m as the closest multiple of 3 smaller than r
        int m = (r_ / 3) * 3; 
        int p = m / 3;
        fdapde_assert(p > 0 && "Hutch++ requires r >= 3");
        //H, G creation and population with rademacher (H corresponds to matrix S in the Hutch++ paper)
        if (!H_.has_value() || H_->rows() != n || H_->cols() != p) {
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            H_ = matrix_t(n, p);
            G_ = matrix_t(n, p);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) { 
                    (*H_)(i, j) = rademacher(rng);
                    (*G_)(i, j) = rademacher(rng);
                }
            }
        }

        matrix_t SH = A.apply(*H_);
        Eigen::ColPivHouseholderQR<matrix_t> qr(SH);
        matrix_t Q = qr.householderQ() * matrix_t::Identity(n, p); //I retain only the first p (m/3) columns of Q
        matrix_t SQ = A.apply(Q);
        // Tr(Q'SQ) element-wise(Hadamard) product
        double tr_QSQ = (Q.cwiseProduct(SQ)).sum();
        // (I - QQ')G
        matrix_t G_p = *G_ - Q * (Q.transpose() * *G_);
        matrix_t SG_p = A.apply(G_p);
        // Tr(G_p'SG_p)
        double tr_GpSGp = (G_p.cwiseProduct(SG_p)).sum();
        trace_est_ = tr_QSQ + (tr_GpSGp / p);
        return;
    }


    // observer
    double trace() const { return trace_est_; }

    private:
    int r_ = 100, seed_ = 0;
    mutable std::optional<matrix_t> H_, G_;
    mutable double trace_est_ = 0.0;

};


template <LinearOperator Op> Hutchpp(const Op&) -> Hutchpp<Op>;
template <LinearOperator Op> Hutchpp(const Op&, int) -> Hutchpp<Op>;
template <LinearOperator Op> Hutchpp(const Op&, int, int) -> Hutchpp<Op>;

} // namespace internals
} // namespace fdapde



#endif //__HUTCHINSONPP_H__