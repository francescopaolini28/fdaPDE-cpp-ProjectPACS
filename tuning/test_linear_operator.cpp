#include <fdaPDE/models.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "linear_operator.h"
#include "hutch.h"
#include "hutchpp.h"
#include "xtrace.h"

using namespace fdapde::internals;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static int n_ok = 0, n_fail = 0;

void check(const char* name, const matrix_t& got, const matrix_t& ref,
           double tol = 1e-10) {
    double err = (got - ref).norm() / std::max(1.0, ref.norm());
    bool ok = err < tol;
    std::cout << (ok ? "[OK]   " : "[FAIL] ")
              << std::left << std::setw(42) << name
              << "rel.err = " << std::scientific << std::setprecision(2) << err << "\n";
    if (ok) ++n_ok; else ++n_fail;
}

// Spectrum generators (from Epperly, Tropp, Webber 2024)
Eigen::VectorXd make_spectrum(int n, const std::string& type) {
    Eigen::VectorXd s(n);
    if (type == "flat") {
        for (int i = 0; i < n; ++i) s(i) = 3.0 - 2.0 * i / (n - 1.0);
    } else if (type == "poly") {
        for (int i = 1; i <= n; ++i) s(i - 1) = 1.0 / (i * i);
    } else if (type == "exp") {
        for (int i = 0; i < n; ++i) s(i) = std::pow(0.7, i);
    } else if (type == "step") {
        for (int i = 0; i < n; ++i) s(i) = (i < 50) ? 1.0 : 1e-3;
    }
    return s;
}

// Build M = Q diag(spec) Q^T with random orthogonal Q (non-diagonal, so
// Hutch has genuine Monte Carlo variance — diagonal would be exact)
Eigen::MatrixXd make_orthogonal_similar(const Eigen::VectorXd& spec, int seed) {
    const int n = spec.size();
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd G(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            G(i, j) = nd(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(G);
    Eigen::MatrixXd Q = qr.householderQ();
    return Q * spec.asDiagonal() * Q.transpose();
}

double effective_rank(const Eigen::VectorXd& spec) {
    double s1 = spec.sum(), s2 = spec.squaredNorm();
    return s1 * s1 / s2;
}

// ---------------------------------------------------------------------------
// Benchmark on one spectrum
// ---------------------------------------------------------------------------
void benchmark(const std::string& spec_name, const Eigen::VectorXd& spec, int seed) {
    const int n = spec.size();
    const int n_trials = 5;
    const std::vector<int> r_values = {10, 20, 50, 100, 200};

    const double true_trS  = spec.sum();
    const double true_trSS = spec.squaredNorm();

    std::cout << "\n================================================================\n"
              << " Spectrum: " << std::left << std::setw(6) << spec_name
              << "   n = " << n << "\n"
              << "================================================================\n"
              << " true Tr[S]     = " << std::setw(14) << true_trS
              << " true Tr[S^T S] = " << true_trSS << "\n"
              << " effective rank = " << std::setw(14) << effective_rank(spec)
              << " cond(S)        = " << spec.maxCoeff() / spec.minCoeff() << "\n\n";

    std::vector<Eigen::MatrixXd> matrices;
    for (int t = 0; t < n_trials; ++t)
        matrices.push_back(make_orthogonal_similar(spec, seed + 1000 * t));

    std::vector<std::array<double,4>> avg_S, avg_SS;

    for (int r : r_values) {
        std::array<double,4> sum_S  = {0,0,0,0};
        std::array<double,4> sum_SS = {0,0,0,0};

        for (int t = 0; t < n_trials; ++t) {
            auto op    = as_dense(matrices[t]);
            auto op_sq = transpose(op) * op;
            int sd = seed + 31 * r + 7 * t;

            double est_S[4] = {
                Hutch     <decltype(op)>(op, r, sd).trace(),
                Hutchpp   <decltype(op)>(op, r, sd).trace(),
                Xtrace    <decltype(op)>(op, r, sd).trace(),
                XNysTrace <decltype(op)>(op, r, sd).trace()
            };
            double est_SS[4] = {
                Hutch     <decltype(op_sq)>(op_sq, r, sd).trace(),
                Hutchpp   <decltype(op_sq)>(op_sq, r, sd).trace(),
                Xtrace    <decltype(op_sq)>(op_sq, r, sd).trace(),
                XNysTrace <decltype(op_sq)>(op_sq, r, sd).trace()
            };

            for (int k = 0; k < 4; ++k) {
                sum_S [k] += std::abs(est_S [k] - true_trS ) / std::abs(true_trS );
                sum_SS[k] += std::abs(est_SS[k] - true_trSS) / std::abs(true_trSS);
            }
        }
        std::array<double,4> a_S, a_SS;
        for (int k = 0; k < 4; ++k) {
            a_S [k] = sum_S [k] / n_trials;
            a_SS[k] = sum_SS[k] / n_trials;
        }
        avg_S .push_back(a_S );
        avg_SS.push_back(a_SS);
    }

    auto print_table = [&](const char* title,
                           const std::vector<std::array<double,4>>& data) {
        std::cout << " ---- " << title << " ----\n"
                  << " " << std::left << std::setw(6) << "r"
                  << std::setw(14) << "Hutch"
                  << std::setw(14) << "Hutch++"
                  << std::setw(14) << "Xtrace"
                  << std::setw(14) << "XNysTrace" << "\n"
                  << " " << std::string(62, '-') << "\n";
        for (size_t i = 0; i < r_values.size(); ++i) {
            std::cout << " " << std::left << std::setw(6) << r_values[i];
            for (int k = 0; k < 4; ++k)
                std::cout << std::setw(14) << std::scientific << std::setprecision(2) << data[i][k];
            std::cout << "\n";
        }
        std::cout << "\n";
    };

    print_table("Tr[S]",     avg_S);
    print_table("Tr[S^T S]", avg_SS);
}

// ---------------------------------------------------------------------------
int main() {
    // =======================================================================
    // SECTION 1 — operator correctness (compact)
    // =======================================================================
    std::cout << "=== 1. OPERATOR CORRECTNESS ===\n";
    const int n = 50, k = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, k);
    Eigen::VectorXd d = (Eigen::VectorXd::Random(n).array().abs() + 0.5).matrix();
    Eigen::MatrixXd U = Eigen::MatrixXd::Random(n, 5);
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(n, 5);
    Eigen::SparseMatrix<double> S = A.sparseView();

    auto opA = as_dense(A);
    check("DenseOp",              opA.apply(X),           A * X);
    check("DenseOp.transpose",    opA.apply_transpose(X), A.transpose() * X);
    check("SparseOp",             as_sparse(S).apply(X),  A * X);

    matrix_t D = matrix_t::Zero(n, n); D.diagonal() = d;
    check("DiagonalOp",           as_diagonal(d).apply(X), D * X);
    check("IdentityOp",           as_identity(n).apply(X), X);
    check("LowRankOp",            as_low_rank(U, V).apply(X), U * V.transpose() * X);

    check("A * B",                (opA * as_dense(B)).apply(X),  A * B * X);
    check("A + B",                (opA + as_dense(B)).apply(X),  (A + B) * X);
    check("A - B",                (opA - as_dense(B)).apply(X),  (A - B) * X);
    check("2.5 * A",              (2.5 * opA).apply(X),          2.5 * A * X);
    check("A^2",                  power<2>(opA).apply(X),        A * A * X);
    check("A^3",                  power<3>(opA).apply(X),        A * A * A * X);
    check("transpose(A) * A",     (transpose(opA) * opA).apply(X),  A.transpose() * A * X);
    check("A^T * B - 2A",         (transpose(opA) * as_dense(B) - 2.0 * opA).apply(X),
                                 (A.transpose() * B - 2.0 * A) * X);
    check("A^T * A + 0.1*I",      (transpose(opA) * opA + 0.1 * as_identity(n)).apply(X),
                                 (A.transpose() * A + 0.1 * matrix_t::Identity(n, n)) * X);

    // inverse
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    auto invA = inverse_operator(lu, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    check("A * (invA * b) - b",   A * invA.apply(b) - b,        matrix_t::Zero(n, 1));

    std::cout << "\n";

    // =======================================================================
    // SECTION 2 — estimator benchmark on fast-decaying spectra
    // =======================================================================
    std::cout << "=== 2. ESTIMATOR BENCHMARK ===\n"
              << " (5 trials per r, averaged relative error)\n";

    const int n_big = 500;
    for (const std::string& name : {"flat", "poly", "exp", "step"}) {
        benchmark(name, make_spectrum(n_big, name), 42);
    }

    // =======================================================================
    // SUMMARY
    // =======================================================================
    std::cout << "\n=== SUMMARY ===\n"
              << "Operator tests:  Passed " << n_ok << ", failed " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}