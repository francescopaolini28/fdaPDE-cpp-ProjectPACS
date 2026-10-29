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
#include <optional>
#include "linear_operator.h"
#include "./fdaPDE/src/models/header_check.h"

namespace fdapde {
namespace internals {

// STochastic trace estimation bu Hutchinson method, as described in:
// M. F. Hutchinson (1989) A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines

template <LinearOperator Op>

class Hutch {

    using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    
    public:

    Hutch() noexcept = default;
    Hutch(const Op& A, int r = 100, int seed = random_seed) : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { compute(A); }
    Hutch(int r, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) { }

    
    void compute(const Op& A) const {
        const int n_ = A.rows();

        if(!Us_.has_value() || Us_->rows() != n_ || Us_->cols() != r_){
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            Us_ = matrix_t(n_, r_);
            for (int i = 0; i < n_; ++i) {
                for (int j = 0; j < r_; ++j) { (*Us_)(i, j) = rademacher(rng); }
            }
        }
        
        matrix_t SU = A.apply(*Us_); 
        trace_est_ = ((*Us_).cwiseProduct(SU)).sum() / r_;  
        return;
    }

    double trace() const { return trace_est_; }


    private:
    int r_ = 100;
    int seed_ = 0;

    mutable std::optional<matrix_t> Us_;
    mutable double trace_est_ = 0.0;

};


// Deduction guide
template <LinearOperator Op>
Hutch(const Op&) -> Hutch<Op>;

template <LinearOperator Op>
Hutch(const Op&, int) -> Hutch<Op>;

template <LinearOperator Op>
Hutch(const Op&, int, int) -> Hutch<Op>;

} // namespace internals
} // namesspace fdapde



#endif //__HUTCHINSON_H__