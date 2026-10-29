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

#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace fdapde {
namespace internals {

using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using vector_t = Eigen::Matrix<double, Eigen::Dynamic, 1>;

// Contract definition. A vector is just an (n x 1) matrix, so apply() always returns a matrix_t
template <typename T>
concept LinearOperator = requires(const T& op, const vector_t& v, const matrix_t& M) {
    { op.rows() } -> std::convertible_to<Eigen::Index>;
    { op.cols() } -> std::convertible_to<Eigen::Index>;
    { op.size() } -> std::convertible_to<Eigen::Index>;
    { op.apply(v) } -> std::convertible_to<matrix_t>; // apply() always returns a (n x 1) matrix_t
    { op.apply(M) } -> std::convertible_to<matrix_t>;
};

// Transponibility
template <typename T>
concept TransposableOperator = LinearOperator<T>
  && requires(const T& op, const vector_t& v, const matrix_t& M) {
      { op.apply_transpose(v) } -> std::convertible_to<matrix_t>;
      { op.apply_transpose(M) } -> std::convertible_to<matrix_t>;
  };


// Dense Operator
template <typename Derived>
class DenseOp {
    const Derived& mat_;
   public:
    explicit DenseOp(const Derived& m) : mat_(m) { }
    explicit DenseOp(const Derived&&) = delete;   
    Eigen::Index rows() const { return mat_.rows(); }
    Eigen::Index cols() const { return mat_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(const Eigen::MatrixBase<X>& x) const {
        return mat_ * x;
    }
    template <typename X> matrix_t apply_transpose(const Eigen::MatrixBase<X>& x) const {
        return mat_.transpose() * x;
    }
};

// Sparse Operator
template <typename Derived>
class SparseOp {
    const Derived& mat_;
   public:
    explicit SparseOp(const Derived& m) : mat_(m) { }
    explicit SparseOp(const Derived&&) = delete;  
    Eigen::Index rows() const { return mat_.rows(); }
    Eigen::Index cols() const { return mat_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(const Eigen::MatrixBase<X>& x) const {
        return mat_ * x;
    }
    template <typename X> matrix_t apply_transpose(const Eigen::MatrixBase<X>& x) const {
        return mat_.transpose() * x;
    }
};

// Identity
class IdentityOp {
    Eigen::Index n_;
   public:
    explicit IdentityOp(Eigen::Index n) : n_(n) { }
    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::Index size() const { return n_ * n_; }
    template <typename X> matrix_t apply(X&& x) const {
        return std::forward<X>(x);
    }
    template <typename X> matrix_t apply_transpose(X&& x) const {
        return std::forward<X>(x);
    }
};

// Diagonal Operator
template <typename Derived>
class DiagonalOp {
    const Derived& d_;
   public:
    explicit DiagonalOp(const Derived& d) : d_(d) { }
    explicit DiagonalOp(const Derived&&) = delete;   
    Eigen::Index rows() const { return d_.size(); }
    Eigen::Index cols() const { return d_.size(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        return d_.asDiagonal() * std::forward<X>(x);
    }
    template <typename X> matrix_t apply_transpose(X&& x) const {
        return d_.asDiagonal() * std::forward<X>(x);   // diagonal = symmetric
    }
};


template <typename ApplyFn, typename ApplyTFn = std::nullptr_t>
class LambdaOp {
    ApplyFn  apply_;
    ApplyTFn apply_t_;
    Eigen::Index rows_, cols_;
   public:
    LambdaOp(ApplyFn f, Eigen::Index r, Eigen::Index c)
      : apply_(std::move(f)), apply_t_(), rows_(r), cols_(c) { }
    LambdaOp(ApplyFn f, ApplyTFn tf, Eigen::Index r, Eigen::Index c)
      : apply_(std::move(f)), apply_t_(std::move(tf)), rows_(r), cols_(c) { }
    Eigen::Index rows() const { return rows_; }
    Eigen::Index cols() const { return cols_; }
    Eigen::Index size() const { return rows_ * cols_; }
    template <typename X> matrix_t apply(X&& x) const {
        return apply_(std::forward<X>(x));
    }
    template <typename X> requires (!std::is_same_v<ApplyTFn, std::nullptr_t>)
    matrix_t apply_transpose(X&& x) const {
        return apply_t_(std::forward<X>(x));
    }
};


// U * V^T, with U (n x r) and V (m x r)
template <typename DerivedU, typename DerivedV>
class LowRankOp {
    const DerivedU& U_;
    const DerivedV& V_;
   public:
    LowRankOp(const DerivedU& U, const DerivedV& V) : U_(U), V_(V) {
        assert(U_.cols() == V_.cols() && "LowRankOp: U and V must have the same number of columns");
    }
    LowRankOp(const DerivedU&&, const DerivedV&) = delete;    // would dangle
    LowRankOp(const DerivedU&, const DerivedV&&) = delete;    // would dangle
    LowRankOp(const DerivedU&&, const DerivedV&&) = delete;   // would dangle
    Eigen::Index rows() const { return U_.rows(); }
    Eigen::Index cols() const { return V_.rows(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(const Eigen::MatrixBase<X>& x) const {
        return U_ * (V_.transpose() * x);
    }
    template <typename X> matrix_t apply_transpose(const Eigen::MatrixBase<X>& x) const {
        return V_ * (U_.transpose() * x);
    }
};


template <typename Solver, bool Symmetric = false>
class InverseOp {
    const Solver& solver_;
    Eigen::Index n_;
   public:
    InverseOp(const Solver& s, Eigen::Index n) : solver_(s), n_(n) { }
    InverseOp(const Solver&&, Eigen::Index) = delete;   // would dangle
    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::Index size() const { return n_ * n_; }
    template <typename X> matrix_t apply(X&& x) const {
        return solver_.solve(std::forward<X>(x));
    }
    template <typename X>
    requires (Symmetric || requires(const Solver& s, X&& y) { s.transpose().solve(std::forward<X>(y)); })
    matrix_t apply_transpose(X&& x) const {
        if constexpr (Symmetric) return solver_.solve(std::forward<X>(x));
        else return solver_.transpose().solve(std::forward<X>(x));
    }
};

template <bool Symmetric = false, typename Solver>
InverseOp<Solver, Symmetric> inverse_operator(const Solver& s, Eigen::Index n) {
    return InverseOp<Solver, Symmetric>(s, n);
}
template <bool Symmetric = false, typename Solver>
void inverse_operator(const Solver&&, Eigen::Index) = delete;   // would dangle


// Null operator (rows x cols)
class NullOp {
    Eigen::Index rows_, cols_;
   public:
    NullOp(Eigen::Index rows, Eigen::Index cols) : rows_(rows), cols_(cols) { }
    explicit NullOp(Eigen::Index n) : NullOp(n, n) { }
    Eigen::Index rows() const { return rows_; }
    Eigen::Index cols() const { return cols_; }
    Eigen::Index size() const { return rows_ * cols_; }
    template <typename X> matrix_t apply(X&& x) const {
        return matrix_t::Zero(rows_, x.cols());
    }
    template <typename X> matrix_t apply_transpose(X&& x) const {
        return matrix_t::Zero(cols_, x.cols());
    }
};

inline NullOp null_operator(Eigen::Index n) { return NullOp(n); }
inline NullOp null_operator(Eigen::Index rows, Eigen::Index cols) { return NullOp(rows, cols); }


// A * B
template <LinearOperator A, LinearOperator B>
class ProductOp {
    A a_; B b_;
   public:
    ProductOp(A a, B b) : a_(std::move(a)), b_(std::move(b)) {
        assert(a_.cols() == b_.rows() && "ProductOp: dimension mismatch");
    }
    Eigen::Index rows() const { return a_.rows(); }
    Eigen::Index cols() const { return b_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        return a_.apply(b_.apply(std::forward<X>(x)));
    }
    template <typename X> requires (TransposableOperator<A> && TransposableOperator<B>)
    matrix_t apply_transpose(X&& x) const {
        return b_.apply_transpose(a_.apply_transpose(std::forward<X>(x)));
    }
};

// A + B
template <LinearOperator A, LinearOperator B>
class SumOp {
    A a_; B b_;
   public:
    SumOp(A a, B b) : a_(std::move(a)), b_(std::move(b)) {
        assert(a_.rows() == b_.rows() && a_.cols() == b_.cols() && "SumOp: dimension mismatch");
    }
    Eigen::Index rows() const { return a_.rows(); }
    Eigen::Index cols() const { return a_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        matrix_t r = a_.apply(x);
        r += b_.apply(x);
        return r;
    }
    template <typename X> requires (TransposableOperator<A> && TransposableOperator<B>)
    matrix_t apply_transpose(X&& x) const {
        matrix_t r = a_.apply_transpose(x);
        r += b_.apply_transpose(x);
        return r;
    }
};

// A - B
template <LinearOperator A, LinearOperator B>
class DifferenceOp {
    A a_; B b_;
   public:
    DifferenceOp(A a, B b) : a_(std::move(a)), b_(std::move(b)) {
        assert(a_.rows() == b_.rows() && a_.cols() == b_.cols() && "DifferenceOp: dimension mismatch");
    }
    Eigen::Index rows() const { return a_.rows(); }
    Eigen::Index cols() const { return a_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        matrix_t r = a_.apply(x);
        r -= b_.apply(x);
        return r;
    }
    template <typename X> requires (TransposableOperator<A> && TransposableOperator<B>)
    matrix_t apply_transpose(X&& x) const {
        matrix_t r = a_.apply_transpose(x);
        r -= b_.apply_transpose(x);
        return r;
    }
};

// s * A
template <LinearOperator A>
class ScaledOp {
    A a_;
    double s_;
   public:
    ScaledOp(A a, double s) : a_(std::move(a)), s_(s) { }
    Eigen::Index rows() const { return a_.rows(); }
    Eigen::Index cols() const { return a_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        matrix_t r = a_.apply(std::forward<X>(x));
        r *= s_;
        return r;
    }
    template <typename X> requires TransposableOperator<A>
    matrix_t apply_transpose(X&& x) const {
        matrix_t r = a_.apply_transpose(std::forward<X>(x));
        r *= s_;
        return r;
    }
};

// A^T
template <TransposableOperator A>
class TransposeOp {
    A a_;
   public:
    explicit TransposeOp(A a) : a_(std::move(a)) { }
    Eigen::Index rows() const { return a_.cols(); }
    Eigen::Index cols() const { return a_.rows(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        return a_.apply_transpose(std::forward<X>(x));
    }
    template <typename X> matrix_t apply_transpose(X&& x) const {
        return a_.apply(std::forward<X>(x));
    }
};

// A^K (K >= 0). K = 0 => Identity
template <LinearOperator A, int K>
    requires (K >= 0)
class PowerOp {
    A a_;
   public:
    explicit PowerOp(A a) : a_(std::move(a)) {
        assert(a_.rows() == a_.cols() && "PowerOp: operator must be square");
    }
    Eigen::Index rows() const { return a_.rows(); }
    Eigen::Index cols() const { return a_.cols(); }
    Eigen::Index size() const { return rows() * cols(); }
    template <typename X> matrix_t apply(X&& x) const {
        matrix_t r = std::forward<X>(x);
        for (int i = 0; i < K; ++i) r = a_.apply(r);
        return r;
    }
    template <typename X> requires TransposableOperator<A>
    matrix_t apply_transpose(X&& x) const {   // (A^K)^T = (A^T)^K
        matrix_t r = std::forward<X>(x);
        for (int i = 0; i < K; ++i) r = a_.apply_transpose(r);
        return r;
    }
};


// factories
template <typename Derived>
DenseOp<Derived> as_dense(const Eigen::MatrixBase<Derived>& m) {
    return DenseOp<Derived>(m.derived());
}
template <typename Derived>
void as_dense(const Eigen::MatrixBase<Derived>&&) = delete;

template <typename Derived>
SparseOp<Derived> as_sparse(const Eigen::SparseMatrixBase<Derived>& m) {
    return SparseOp<Derived>(m.derived());
}
template <typename Derived>
void as_sparse(const Eigen::SparseMatrixBase<Derived>&&) = delete;

template <typename Derived>
DiagonalOp<Derived> as_diagonal(const Eigen::MatrixBase<Derived>& d) {
    return DiagonalOp<Derived>(d.derived());
}
template <typename Derived>
void as_diagonal(const Eigen::MatrixBase<Derived>&&) = delete;

template <typename Fn>
LambdaOp<Fn> as_lambda(Fn f, Eigen::Index r, Eigen::Index c) {
    return LambdaOp<Fn>(std::move(f), r, c);
}
template <typename Fn, typename TFn>
LambdaOp<Fn, TFn> as_lambda(Fn f, TFn tf, Eigen::Index r, Eigen::Index c) {
    return LambdaOp<Fn, TFn>(std::move(f), std::move(tf), r, c);
}
inline IdentityOp as_identity(Eigen::Index n) { return IdentityOp(n); }

template <typename DerivedU, typename DerivedV>
LowRankOp<DerivedU, DerivedV>
as_low_rank(const Eigen::MatrixBase<DerivedU>& U, const Eigen::MatrixBase<DerivedV>& V) {
    return LowRankOp<DerivedU, DerivedV>(U.derived(), V.derived());
}

template <typename U, typename V>
    requires (!std::is_lvalue_reference_v<U> || !std::is_lvalue_reference_v<V>)
void as_low_rank(U&&, V&&) = delete;

template <TransposableOperator A>
TransposeOp<A> transpose(A a) {
    return TransposeOp<A>(std::move(a));
}

template <int K, LinearOperator A>
PowerOp<A, K> power(A a) {
    return PowerOp<A, K>(std::move(a));
}


template <LinearOperator A, LinearOperator B>
ProductOp<A, B> operator*(A a, B b) {
    return ProductOp<A, B>(std::move(a), std::move(b));
}

template <LinearOperator A, LinearOperator B>
SumOp<A, B> operator+(A a, B b) {
    return SumOp<A, B>(std::move(a), std::move(b));
}

template <LinearOperator A, LinearOperator B>
DifferenceOp<A, B> operator-(A a, B b) {
    return DifferenceOp<A, B>(std::move(a), std::move(b));
}

template <LinearOperator A>
ScaledOp<A> operator*(A a, double s) {
    return ScaledOp<A>(std::move(a), s);
}
template <LinearOperator A>
ScaledOp<A> operator*(double s, A a) {
    return ScaledOp<A>(std::move(a), s);
}

//debugging only
template <LinearOperator A>
matrix_t materialize(const A& op) {
    const Eigen::Index n = op.cols();
    matrix_t I = matrix_t::Identity(n, n);
    return op.apply(I);
}

}   // namespace internals
}   // namespace fdapde

#endif // __LINEAR_OPERATOR_H__