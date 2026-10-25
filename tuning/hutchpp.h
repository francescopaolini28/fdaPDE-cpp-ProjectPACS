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

namespace fdapde {
namespace internals {

// STochastic trace estimation bu Hutchinson method, as described in:
// M. F. Hutchinson (1989) A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines

template <typename MatrixType>
class Hutchpp {
    
    public:

    Hutchpp() noexcept = default;
    Hutchpp(int n, int r = 100, int seed = random_seed) noexcept : n_(n), r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

    template <typename SolverType>
    void compute_S(SolverType& solver) const {
        // fdapde_assert(solver.lambda_saved_.has_value());  (da mettere in fe_ls_elliptic)
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        // Defining m as the closest multiple of 3 smaller than r
        int m = (r_ / 3) * 3; 
        int p = m / 3;
        //H, G creation and population with rademacher (H corresponds to matrix S in the Hutch++ paper)
        MatrixType H(n_, p);
        MatrixType G(n_, p);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < p; ++j) { 
                H(i, j) = rademacher(rng);
                G(i, j) = rademacher(rng);
            }
        }
        MatrixType SH = solver.apply_S(H);
        Eigen::ColPivHouseholderQR<MatrixType> qr(SH);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, p); //I retain only the first p (m/3) columns of Q
        MatrixType SQ = solver.apply_S(Q);
        // Tr(Q'SQ) element-wise(Hadamard) product
        double tr_QSQ = (Q.cwiseProduct(SQ)).sum();
        // (I - QQ')G
        MatrixType G_p = G - Q * (Q.transpose() * G);
        MatrixType SG_p = solver.apply_S(G_p);
        // Tr(G_p'SG_p)
        double tr_GpSGp = (G_p.cwiseProduct(SG_p)).sum();
        Strace_est_ = tr_QSQ + (tr_GpSGp / p);
        return;
    }

    template <typename SolverType>
    void compute_SS(SolverType& solver) const {
            
        // fdapde_assert(solver.lambda_saved_.has_value());  (da mettere in fe_ls_elliptic)
        std::mt19937 rng(seed_);
        rademacher_distribution rademacher;
        // Defining m as the closest multiple of 3 smaller than r
        int m = (r_ / 3) * 3; 
        int p = m / 3;
        //H, G creation and population with rademacher (H corresponds to matrix S in the Hutch++ paper)
        MatrixType H(n_, p);
        MatrixType G(n_, p);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < p; ++j) { 
                H(i, j) = rademacher(rng);
                G(i, j) = rademacher(rng);
            }
        }
        // S'SH
        MatrixType SSH = solver.apply_St(solver.apply_S(H)); //n_locs x p matrix, In the paper SSH = AS
        Eigen::ColPivHouseholderQR<MatrixType> qr(SSH);
        MatrixType Q = qr.householderQ() * MatrixType::Identity(n_, p); //I retain only the first p (m/3) columns of Q
        // tr(Q'SSQ)
        MatrixType SQ = solver.apply_S(Q);
        // always Frobenius norm on S'S
        double tr_QSSQ = SQ.squaredNorm();
        // (I - QQ')G 
        MatrixType G_p = G - Q * (Q.transpose() * G);
        MatrixType SG_p = solver.apply_S(G_p);
        double tr_GpSSGp = SG_p.squaredNorm();
        SStrace_est_ =  tr_QSSQ + (tr_GpSSGp / p); 
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



#endif //__HUTCHINSONPP_H__