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

#ifndef __XTRACE_H__
#define __XTRACE_H__

#include <random>
#include <optional>
#include <Eigen/Dense>
#include "linear_operator.h"
#include "./fdaPDE/src/models/header_check.h"


namespace fdapde {
namespace internals {

// Stochastic trace estimation bu Xtrace and Xnystrace methods, as described in:
// Ethan N. Epperly, Joel A. Tropp, Robert J. Webber (2024) XTRACE: Making the most of every sample in stochastic trace estimation

template <LinearOperator Op>
class Xtrace{
    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    public:

    Xtrace() noexcept = default;
    Xtrace(const Op& A, int r = 100, int seed = random_seed) : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { compute(A); }
    Xtrace(int r, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { }


    void compute(const Op& A) const {
        const int n = A.rows();
        const int m = (r_ / 2) * 2; 
        const int p = m / 2;
        fdapde_assert(p > 0 && "Xtrace requires r >= 2");

        if (!O_.has_value() || O_->rows() != n || O_->cols() != p) {

            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            O_ = matrix_t(n, p);
            for (int i = 0; i < n; ++i){
                for (int j = 0; j < p; ++j){
                    (*O_)(i, j) = rademacher(rng);
                }
            }       
            
        
        
        }

        matrix_t Y = A.apply(*O_);
        Eigen::HouseholderQR<matrix_t> qr(Y);
        matrix_t Q = qr.householderQ() * matrix_t::Identity(n, p);
        matrix_t R = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        matrix_t Z = A.apply(Q);
        matrix_t H = Q.transpose() * Z;
        matrix_t W = Q.transpose() * *O_;
        matrix_t T = Z.transpose() * *O_;
        matrix_t S = R.inverse().transpose();
        // Normalizing S
        for (int i = 0; i < p; i++) {S.col(i).normalize(); } //each element is normalized by the L2 norm of its column
        //Estimating The trace
        double trace_H = H.trace();
        double trace = 0.0;
        for (int i = 0; i < p; i++) {
            vector_t w_i = W.col(i);
            vector_t s_i = S.col(i);
            vector_t r_i = R.col(i);
            vector_t t_i = T.col(i);
            vector_t x_i = w_i - s_i.dot(w_i) * s_i;
            trace += trace_H - s_i.dot((H*s_i)) + w_i.dot(s_i)*s_i.dot(r_i) - t_i.dot(x_i) + x_i.dot(H*x_i);
        }
        trace_est_ = trace/p;
        return;
    }

    // observer
    double trace() const { return trace_est_; }
    

    private:
    int r_ = 100, seed_ = 0;
    mutable std::optional<matrix_t> O_;
    mutable double trace_est_ = 0.0;
};


template <LinearOperator Op> 
Xtrace(const Op&) -> Xtrace<Op>;
template <LinearOperator Op> 
Xtrace(const Op&, int) -> Xtrace<Op>;
template <LinearOperator Op> 
Xtrace(const Op&, int, int) -> Xtrace<Op>;


//XNysTrace
template <LinearOperator Op>
class XNysTrace{
    
    public:

    XNysTrace() noexcept = default;
    XNysTrace(const Op& A, int r = 100, int seed = random_seed) : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { compute(A); }
    XNysTrace(int r, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { }

    void compute(const Op& A) const {
        const int n = static_cast<int>(A.rows());
        const int m = r_;
        fdapde_assert(m > 0 && "XNysTrace requires r >= 1");

        if (!O_.has_value() || O_->rows() != n || O_->cols() != m) {
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            O_ = matrix_t(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    (*O_)(i, j) = rademacher(rng);
        }

        matrix_t Y = A.apply(*O_);
        const double nu = std::numeric_limits<double>::epsilon() * Y.norm()
                        / std::sqrt(static_cast<double>(n));
        Y += nu * (*O_);

        Eigen::HouseholderQR<matrix_t> qr(Y);
        matrix_t Q = qr.householderQ() * matrix_t::Identity(n, m);
        matrix_t R = qr.matrixQR().topRows(m).template triangularView<Eigen::Upper>();

        matrix_t H = (*O_).transpose() * Y;
        H = 0.5 * (H + H.transpose());
        Eigen::LLT<matrix_t> llt(H);
        matrix_t C = llt.matrixL().transpose();
        matrix_t Cinv = C.template triangularView<Eigen::Upper>()
                            .solve(matrix_t::Identity(m, m));
        matrix_t B = R * Cinv;

        Eigen::HouseholderQR<matrix_t> qrO(*O_);
        matrix_t QQ = qrO.householderQ() * matrix_t::Identity(n, m);
        matrix_t RR = qrO.matrixQR().topRows(m).template triangularView<Eigen::Upper>();


        matrix_t WW = QQ.transpose() * *O_;
        matrix_t RRinvT = RR.template triangularView<Eigen::Upper>()
                              .solve(matrix_t::Identity(m, m)).transpose();

        matrix_t SS = RRinvT;
        for (int i = 0; i < m; ++i) SS.col(i).normalize();

        std::vector<double> scale(m);
        for (int i = 0; i < m; ++i) {
            double norm_ww_i = WW.col(i).norm();
            double d_i       = SS.col(i).dot(WW.col(i));
            scale[i] = (n - m + 1.0) /
                       (n - norm_ww_i * norm_ww_i + std::pow(std::abs(d_i), 2));
        }

        matrix_t W = Q.transpose() * *O_;
        matrix_t Hinv = llt.solve(matrix_t::Identity(m, m));
        matrix_t BCinvT = B * Cinv.transpose();

        matrix_t S_mat(m, m);
        for (int i = 0; i < m; ++i)
            S_mat.col(i) = BCinvT.col(i) * (1.0 / std::sqrt(Hinv(i, i)));

        const double normB2 = B.squaredNorm();
        double trace_est = 0.0;
        for (int i = 0; i < m; ++i) {
            double dSW_i    = S_mat.col(i).dot(W.col(i));
            double normS2_i = S_mat.col(i).squaredNorm();
            trace_est += normB2 - normS2_i
                       + std::pow(std::abs(dSW_i), 2) * scale[i]
                       - nu * n;
        }
        trace_est_ = trace_est / static_cast<double>(m);
    }

    

    // observer
    double trace() const { return trace_est_; }

    private:
        int r_ = 100, seed_ = 0;
        mutable std::optional<matrix_t> O_;
        mutable double trace_est_ = 0.0;
};

template <LinearOperator Op> 
XNysTrace(const Op&) -> XNysTrace<Op>;
template <LinearOperator Op> 
XNysTrace(const Op&, int) -> XNysTrace<Op>;
template <LinearOperator Op> 
XNysTrace(const Op&, int, int) -> XNysTrace<Op>;

} // namespace internals
} // namespace fdapde



#endif //__XTRACE_H__