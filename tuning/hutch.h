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

#ifndef __HUTCHINSON_H__
#define __HUTCHINSON_H__

#include <random>
#include <Eigen/Dense>

namespace fdapde {
namespace internals {

// STochastic trace estimation bu Hutchinson method, as described in:
// M. F. Hutchinson (1989) A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines

template <MatrixType>
class Hutch {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    
    public:

    Hutch(int r = 100, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

    template <typename SolverType>
    double compute_S(SolverType& solver) const {
        fdapde_assert(solver.lambda_saved_.has_value());
        if (!solver.Ys_.has_value() || !solver.Bs_.has_value()) {
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            solver.Us_ = matrix_t(solver.n_locs_, r_);
            for (int i = 0; i < solver.n_locs_; ++i) {
                // sampling from a Rademacher distribution
                for (int j = 0; j < r_; ++j) { solver.Us_->operator()(i, j) = rademacher(rng); }
            }
            //construction of Y
            solver.Ys_ = solver.Us_->transpose() * solver.Psi_;
            solver.Bs_ = matrix_t::Zero(2 * solver.n_dofs_, r_);   // implicitly enforce homogeneous forcing
        }
        // Construction of Bs
        if (solver.n_covs_ == 0) {
            solver.Bs_->topRows(solver.n_dofs_) = -solver.PsiNA().transpose() * solver.D_ * solver.W_ * (*solver.Us_);
        } else {
            solver.Bs_->topRows(solver.n_dofs_) = -solver.PsiNA().transpose() * solver.D_ * internals::lmbQ(solver.W_, solver.X_, solver.invXtWX_, *solver.Us_);
        }
        // enforce Dirichlet BCs, if any
        for (size_t i = 0; i < solver.dirichlet_dofs_.size(); ++i) {
            solver.Bs_->row(solver.dirichlet_dofs_[i]).setConstant(solver.dirichlet_vals_[i]);
        }
        //SMW decomposition to solve MsX = Bs
        // Approximately O(N) due to the sparsity pattern of A, invA_ is Ms
        matrix_t x = solver.n_covs_ == 0 ? solver.invA_.solve(*solver.Bs_) : woodbury_system_solve(solver.invA_, solver.U_, solver.XtWX_, solver.V_, *solver.Bs_); 
        double trS = 0;   // monte carlo Tr[S] approximation
        for (int i = 0; i < r_; ++i) { trS += solver.Ys_->row(i).dot(x.col(i).head(solver.n_dofs_)); }
        return trS / r_;
    }


    template <typename SolverType>
    double compute_SS(SolverType& solver) const {
        
        fdapde_assert(solver.lambda_saved_.has_value());
        if (!solver.Bs_.has_value()) {
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            solver.Us_ = matrix_t(solver.n_locs_, r_);
            for (int i = 0; i < solver.n_locs_; ++i) {
                for (int j = 0; j < r_; ++j) { solver.Us_->operator()(i, j) = rademacher(rng); }
            }
            // Bs_ = Us_ projected onto the mesh space
            solver.Bs_ = matrix_t::Zero(2 * solver.n_dofs_, r_);
        }
            
        // Populating Bs_ (D_ and W_ take account for the weigths given to data and areal data. To be checked)
        if (solver.n_covs_ == 0) {
            solver.Bs_->topRows(solver.n_dofs_) = -solver.PsiNA().transpose() * solver.D_ * solver.W_ * (*solver.Us_);
        } else {
            solver.Bs_->topRows(solver.n_dofs_) = -solver.PsiNA().transpose() * solver.D_ * internals::lmbQ(solver.W_, solver.X_, solver.invXtWX_, *solver.Us_);
        }
        // enforce Dirichlet BCs, if any
        for (size_t i = 0; i < solver.dirichlet_dofs_.size(); ++i) {
            solver.Bs_->row(solver.dirichlet_dofs_[i]).setConstant(solver.dirichlet_vals_[i]);
        }
        // Solving for f_sim (stored in matrix x)
        matrix_t x = solver.n_covs_ == 0 ? solver.invA_.solve(*solver.Bs_) : woodbury_system_solve(solver.invA_, solver.U_, solver.XtWX_, solver.V_, *solver.Bs_);
            
        double trStS = 0.0;
        for (int i = 0; i < r_; ++i) {
            vector_t f_sim = x.col(i).head(solver.n_dofs_);
            if (solver.n_covs_ == 0) {
                // y_hat = S * u_i = Psi_ * f_sim 
                vector_t y_hat = solver.Psi_ * f_sim;
                trStS += y_hat.squaredNorm();
            } else {
                vector_t beta_sim = solver.invXtWXXtW_ * (solver.Us_->col(i) - solver.Psi_ * f_sim);
                vector_t y_hat = solver.Psi_ * f_sim + solver.X_ * beta_sim;
                trStS += y_hat.squaredNorm();
            }
        }
        return trStS / r_;
    }


    private:
    int r_, seed_;


}



} // namespace internals
} // namesspace fdapde



#endif //__HUTCHINSON_H__