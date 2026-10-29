#include <iostream>
#include <cmath>
#include <cassert>
#include "linear_operator.h"

using namespace fdapde::internals;

static int tests_passed = 0;
static int tests_failed = 0;

template <typename Op>
void check_vec(const Op& op, const Eigen::VectorXd& M_x, const Eigen::VectorXd& x, const char* name) {
    const double tol = 1e-10;
    Eigen::VectorXd got = op.apply(x);
    double err = (got - M_x).norm() / std::max(1.0, M_x.norm());
    bool ok = err < tol;
    std::cout << (ok ? "[OK]   " : "[FAIL] ") << name << " (vec)"
              << "  (err = " << err << ")\n";
    if (ok) ++tests_passed; else ++tests_failed;
}

template <typename Op>
void check_mat(const Op& op, const matrix_t& M_X, const matrix_t& X, const char* name) {
    const double tol = 1e-10;
    matrix_t got = op.apply(X);
    double err = (got - M_X).norm() / std::max(1.0, M_X.norm());
    bool ok = err < tol;
    std::cout << (ok ? "[OK]   " : "[FAIL] ") << name << " (mat)"
              << "  (err = " << err << ")\n";
    if (ok) ++tests_passed; else ++tests_failed;
}

int main() {
    const int n = 40;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
    Eigen::VectorXd x = Eigen::VectorXd::Random(n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, 7);
    Eigen::VectorXd d = (Eigen::VectorXd::Random(n).array().abs() + 0.1).matrix();

    std::cout << "=== Leaves ===\n";
    {
        auto op = make_dense_op(A);
        check_vec(op, A * x, x, "DenseOp");
        check_mat(op, A * X, X, "DenseOp");
    }
    {
        Eigen::SparseMatrix<double> S = A.sparseView();
        auto op = make_sparse_op(S);
        check_vec(op, A * x, x, "SparseOp");
    }
    {
        auto op = make_identity_op(n);
        check_vec(op, x, x, "IdentityOp");
    }
    {
        auto op = make_diagonal_op(d);
        check_vec(op, d.asDiagonal() * x, x, "DiagonalOp");
    }
    {
        Eigen::MatrixXd U = Eigen::MatrixXd::Random(n, 5);
        Eigen::MatrixXd V = Eigen::MatrixXd::Random(n, 5);
        auto op = make_low_rank_op(U, V);
        check_vec(op, U * (V.transpose() * x), x, "LowRankOp");
        check_mat(op, U * (V.transpose() * X), X, "LowRankOp");
    }

    std::cout << "\n=== Composizioni (2 operatori) ===\n";
    auto opA = make_dense_op(A);
    auto opB = make_dense_op(B);

    {
        auto op = opA * opB;
        check_vec(op, A * B * x, x, "A*B");
        check_mat(op, A * B * X, X, "A*B");
    }
    {
        auto op = opA + opB;
        check_vec(op, (A + B) * x, x, "A+B");
    }
    {
        auto op = opA - opB;
        check_vec(op, (A - B) * x, x, "A-B");
    }
    {
        auto op = 2.5 * opA;
        check_vec(op, 2.5 * A * x, x, "2.5*A");
    }
    {
        auto op = transpose(opA);
        check_vec(op, A.transpose() * x, x, "A^T");
        check_vec(transpose(op), A * x, x, "(A^T)^T");
    }
    {
        auto op = opA * transpose(opA);
        check_vec(op, A * A.transpose() * x, x, "A*A^T");
        check_mat(op, A * A.transpose() * X, X, "A*A^T");
    }
    {
        auto op = transpose(opA) * opA;
        check_vec(op, A.transpose() * A * x, x, "A^T*A");
        check_mat(op, A.transpose() * A * X, X, "A^T*A");
    }

    {
        auto S = make_dense_op(A);
        auto SqS = transpose(S) * S;
        check_vec(SqS, A.transpose() * A * x, x, "transpose(S)*S");
        check_mat(SqS, A.transpose() * A * X, X, "transpose(S)*S");
    }

    std::cout << "\n=== Riepilogo ===\n"
              << "Passed: " << tests_passed << "\n"
              << "Failed: " << tests_failed << "\n";

    return tests_failed == 0 ? 0 : 1;
}