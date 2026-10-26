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
#include <Eigen/Dense>

namespace fdapde {
namespace internals {

// Stochastic trace estimation bu Xtrace and Xnystrace methods, as described in:
// Ethan N. Epperly, Joel A. Tropp, Robert J. Webber (2024) XTRACE: Making the most of every sample in stochastic trace estimation

template <typename MatrixType>
class Xtrace{
    
    using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    public:

    Xtrace() noexcept = default;
    Xtrace(int n, int r = 100, int seed = random_seed) noexcept : n_(n), r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

    template <typename SolverType>
    void compute_S(SolverType& solver) const {
        // fdapde_assert(solver.lambda_saved_.has_value());
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        //Making sure m is an even integer (Xtrace requirement)
        int m = (r_ / 2) * 2; 
        int p = m / 2;
        MatrixType O(n_, p);
        MatrixType Y(n_, p);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < p; ++j) { 
                O(i, j) = rademacher(rng);
            }
        }

        Y = solver.apply_S(O);
        Eigen::HouseholderQR<MatrixType> qr(Y);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, p);
        MatrixType R = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        MatrixType Z = solver.apply_S(Q);
        MatrixType H = Q.transpose() * Z;
        MatrixType W = Q.transpose() * O;
        MatrixType T = Z.transpose() * O;
        MatrixType S = R.inverse().transpose();
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
        Strace_est_ = trace/p;
        return;
    }

    template <typename SolverType>
    void compute_SS(SolverType& solver) const {
        // fdapde_assert(solver.lambda_saved_.has_value());
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        //Making sure m is an even integer (Xtrace requirement)
        int m = (r_ / 2) * 2; 
        int p = m / 2;
        MatrixType O(n_, p);
        MatrixType Y(n_, p);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < p; ++j) { 
                O(i, j) = rademacher(rng);
            }
        }

        Y = solver.apply_St(solver.apply_S(O));
        Eigen::HouseholderQR<MatrixType> qr(Y);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, p);
        MatrixType R = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        MatrixType Z = solver.apply_St(solver.apply_S(Q));
        MatrixType H = Q.transpose() * Z;
        MatrixType W = Q.transpose() * O;
        MatrixType T = Z.transpose() * O;
        MatrixType S = R.inverse().transpose();
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
        SStrace_est_ = trace/p;
        return;
    }

    // observers
    double S_trace() const { return Strace_est_; }
    double SS_trace() const { return SStrace_est_; }

    private:
    int r_ = 100;
    int seed_ = 0;
    int n_ = 0; //matrix dimension 

    mutable double Strace_est_ = 0.0;
    mutable double SStrace_est_ = 0.0;
};


template <typename MatrixType>
class XNysTrace{
    
    public:

    XNysTrace() noexcept = default;
    XNysTrace(int n, int r = 100, int seed = random_seed) noexcept : n_(n), r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

    template <typename SolverType>
    void compute_S(SolverType& solver) const {
        // fdapde_assert(solver.lambda_saved_.has_value());
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        MatrixType O(n_, r_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < r_; ++j) {
                O(i, j) = rademacher(rng);
            }
        }

        MatrixType Y = solver.apply_S(O);
        double nu = std::numeric_limits<double>::epsilon() * Y.norm() / std::sqrt(double(n_));
        Y += nu * O;
        Eigen::HouseholderQR<MatrixType> qr(Y);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, r_);
        MatrixType R = qr.matrixQR().topRows(r_).template triangularView<Eigen::Upper>();
        MatrixType H = O.transpose() * Y;
        H = 0.5 * (H + H.transpose());
        Eigen::LLT<MatrixType> llt(H);
        MatrixType C = llt.matrixL().transpose(); 
        MatrixType Cinv = C.template triangularView<Eigen::Upper>().solve(MatrixType::Identity(r_, r_));
        MatrixType B = R * Cinv;
        Eigen::HouseholderQR<MatrixType> qrO(O);
        MatrixType QQ = qrO.householderQ() * MatrixType::Identity(n_, r_);
        MatrixType RR = qrO.matrixQR().topRows(r_).template triangularView<Eigen::Upper>();
        MatrixType WW = QQ.transpose() * O;
        MatrixType RRinvT = RR.template triangularView<Eigen::Upper>().solve(MatrixType::Identity(r_, r_)).transpose();
        
        MatrixType SS = RRinvT;
        for(int i = 0; i < r_; ++i) { SS.col(i).normalize(); }

        std::vector<double> scale(r_);
        for(int i = 0; i < r_; ++i) {
            double norm_ww_i = WW.col(i).norm();
            double d_i = SS.col(i).dot(WW.col(i)); 
            scale[i] = (n_ - r_ + 1.0) / (n_ - norm_ww_i * norm_ww_i + std::pow(std::abs(d_i), 2));
        }

        MatrixType W = Q.transpose() * O;
        MatrixType Hinv = llt.solve(MatrixType::Identity(r_, r_)); 
        MatrixType BCinvT = B * Cinv.transpose();
        MatrixType S_mat(r_, r_);
        for(int i = 0; i < r_; ++i) { S_mat.col(i) = BCinvT.col(i) * (1.0 / std::sqrt(Hinv(i, i))); }

        double normB2 = B.squaredNorm();
        double trace = 0.0;

        for(int i = 0; i < r_; ++i) {
            double dSW_i = S_mat.col(i).dot(W.col(i));
            double normS2_i = S_mat.col(i).squaredNorm();
            double est_i = normB2 - normS2_i + std::pow(std::abs(dSW_i), 2) * scale[i] - nu * n_;
            trace += est_i;
        }

        Strace_est_ = trace / r_;
        return;
    }

    template <typename SolverType>
    void compute_SS(SolverType& solver) const {
        // fdapde_assert(solver.lambda_saved_.has_value());
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        MatrixType O(n_, r_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < r_; ++j) {
                O(i, j) = rademacher(rng);
            }
        }

        
        MatrixType Y = solver.apply_St(solver.apply_S(O));
        double nu = std::numeric_limits<double>::epsilon() * Y.norm() / std::sqrt(double(n_));
        Y += nu * O;
        Eigen::HouseholderQR<MatrixType> qr(Y);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, r_);
        MatrixType R = qr.matrixQR().topRows(r_).template triangularView<Eigen::Upper>();

        MatrixType H = O.transpose() * Y;
        H = 0.5 * (H + H.transpose());
        Eigen::LLT<MatrixType> llt(H);
        MatrixType C = llt.matrixL().transpose();
        MatrixType Cinv = C.template triangularView<Eigen::Upper>().solve(MatrixType::Identity(r_, r_));
        MatrixType B = R * Cinv;
        Eigen::HouseholderQR<MatrixType> qrO(O);
        MatrixType QQ = qrO.householderQ() * MatrixType::Identity(n_, r_);
        MatrixType RR = qrO.matrixQR().topRows(r_).template triangularView<Eigen::Upper>();
        MatrixType WW = QQ.transpose() * O;
        MatrixType RRinvT = RR.template triangularView<Eigen::Upper>().solve(MatrixType::Identity(r_, r_)).transpose();
        MatrixType SS = RRinvT;
        for(int i = 0; i < r_; ++i) {
            SS.col(i).normalize();
        }

        std::vector<double> scale(r_);
        for(int i = 0; i < r_; ++i) {
            double norm_ww_i = WW.col(i).norm();
            double d_i = SS.col(i).dot(WW.col(i)); 
            scale[i] = (n_ - r_ + 1.0) / (n_ - norm_ww_i * norm_ww_i + std::pow(std::abs(d_i), 2));
        }

        MatrixType W = Q.transpose() * O;
        MatrixType Hinv = llt.solve(MatrixType::Identity(r_, r_));
        MatrixType BCinvT = B * Cinv.transpose();
        MatrixType S_mat(r_, r_);
        for(int i = 0; i < r_; ++i) {
            S_mat.col(i) = BCinvT.col(i) * (1.0 / std::sqrt(Hinv(i, i)));
        }

        double normB2 = B.squaredNorm();
        double trace = 0.0;
        
        for(int i = 0; i < r_; ++i) {
            double dSW_i = S_mat.col(i).dot(W.col(i));
            double normS2_i = S_mat.col(i).squaredNorm();
            double est_i = normB2 - normS2_i + std::pow(std::abs(dSW_i), 2) * scale[i] - nu * n_;
            trace += est_i;
        }

        SStrace_est_ = trace / r_;
        return;
    }

    // observers
    double S_trace() const { return Strace_est_; }
    double SS_trace() const { return SStrace_est_; }

    private:
    int r_ = 100;
    int seed_ = 0;
    int n_ = 0; //matrix dimension 

    mutable double Strace_est_ = 0.0;
    mutable double SStrace_est_ = 0.0;
};



} // namespace internals
} // namespace fdapde



#endif //__XTRACE_H__