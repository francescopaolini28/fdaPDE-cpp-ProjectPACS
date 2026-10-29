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

#ifndef __LINEAR_OPERATOR_H__
#define __LINEAR_OPERATOR_H__

#include <concepts>
#include <type_traits>
#include <utility>
#include <Eigen/Dense>

namespace fdapde {
namespace internals {

using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

// Contract: rows(), cols(), size(), apply(v), apply(M)
template <typename T>
concept LinearOperator = requires(const T& op) {
    { op.rows() } -> std::convertible_to<int>;
    { op.cols() } -> std::convertible_to<int>;
    { op.size() } -> std::convertible_to<int>;
} && requires(const T& op, const vector_t& v, const matrix_t& M) {
    { op.apply(v) } -> std::convertible_to<vector_t>;
    { op.apply(M) } -> std::convertible_to<matrix_t>;
};

// Stronger contract: also apply_transpose(v), apply_transpose(M)
template <typename T>
concept TransposableOperator = LinearOperator<T>
  && requires(const T& op, const vector_t& v, const matrix_t& M) {
      { op.apply_transpose(v) } -> std::convertible_to<vector_t>;
      { op.apply_transpose(M) } -> std::convertible_to<matrix_t>;
  };


template <TransposableOperator A>
class TransposeOp {
    A a_;
   public:
    explicit TransposeOp(A a) : a_(std::move(a)) { }
    int rows() const { return a_.cols(); }
    int cols() const { return a_.rows(); }
    int size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        return a_.apply_transpose(std::forward<X>(x));
    }
    template <typename X> matrix_t apply_transpose(X&& x) const {
        return a_.apply(std::forward<X>(x));
    }
};


template <LinearOperator A, LinearOperator B>
class ProductOp {
    A a_; B b_;
   public:
    ProductOp(A a, B b) : a_(std::move(a)), b_(std::move(b)) { }
    int rows() const { return a_.rows(); }
    int cols() const { return b_.cols(); }
    int size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        return a_.apply(b_.apply(std::forward<X>(x)));
    }
    template <typename X> requires (TransposableOperator<A> && TransposableOperator<B>)
    matrix_t apply_transpose(X&& x) const {
        return b_.apply_transpose(a_.apply_transpose(std::forward<X>(x)));
    }
};


template <TransposableOperator A>
TransposeOp<A> transpose(A a) {
    return TransposeOp<A>(std::move(a));
}

template <LinearOperator A, LinearOperator B>
ProductOp<A, B> operator*(A a, B b) {
    return ProductOp<A, B>(std::move(a), std::move(b));
}

}   // namespace internals
}   // namespace fdapde

#endif // __LINEAR_OPERATOR_H__