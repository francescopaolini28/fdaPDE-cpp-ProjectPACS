// Benchmark degli stimatori di traccia di fe_ls_elliptic (Hutch, Hutchpp, Xtrace, XNysTrace)
// Le quattro struct sono COPIATE VERBATIM dal solver. Il MockSolver espone (pubblici) i soli membri
// che esse usano, con S = A (matrice densa con spettro noto), cosi' la traccia vera e' sum(lambda).
//
// Compilazione:
//   g++ -O3 -std=c++20 -march=native -fopenmp -I /path/to/eigen benchmark_trace_mock.cpp -o bench
// Uso:
//   ./bench [n_trials=100] [N=1000]
//
// Output:
//   benchmark_trace_raw.csv      una riga per (trial, matrice, metodo, m)
//   benchmark_trace_summary.csv  errore relativo medio sui trial (pronto per il plot)

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <cstdlib>

using namespace Eigen;
using namespace std::chrono;

// ------------------------------------------------------------------------------------
// Dipendenze "di libreria" richieste dalle struct (definizioni minime per il benchmark)
// ------------------------------------------------------------------------------------
constexpr int random_seed = -1;   // sentinella usata come default dei costruttori (qui sempre passiamo un seed)
#define fdapde_assert(cond) do { if (!(cond)) throw std::runtime_error("assert fallita: " #cond); } while (0)

struct rademacher_distribution {
    std::uniform_int_distribution<int> dist{0, 1};
    template <typename Generator>
    double operator()(Generator& g) { return dist(g) * 2.0 - 1.0; }
};

// Rami con covariate (n_covs_ != 0): non usati qui (n_covs_ = 0), ma i nomi devono esistere per compilare.
namespace internals {
template <typename... Args> Eigen::MatrixXd lmbQ(Args&&...) { throw std::runtime_error("lmbQ non disponibile nel mock"); }
}
template <typename... Args> Eigen::MatrixXd woodbury_system_solve(Args&&...) {
    throw std::runtime_error("woodbury_system_solve non disponibile nel mock");
}

// "invA_" del mock: risolve il sistema a blocchi in modo che, con Psi = I, D = I, W = I:
//   rhs = [ -u ; 0 ]  ->  x = [ S*u ; 0 ]   con S = A
struct MockInvA {
    const Eigen::MatrixXd* A = nullptr;
    template <typename Rhs> Eigen::MatrixXd solve(const Rhs& rhs) const {
        const int n = A->rows();
        Eigen::MatrixXd x = Eigen::MatrixXd::Zero(2 * n, rhs.cols());
        x.topRows(n) = -(*A) * rhs.topRows(n);
        return x;
    }
};

// ------------------------------------------------------------------------------------
// MockSolver: stessa interfaccia (membri e metodi) usata dagli stimatori
// ------------------------------------------------------------------------------------
struct MockSolver {
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using sparse_matrix_t = Eigen::SparseMatrix<double>;
    using diag_matrix_t   = Eigen::DiagonalMatrix<double, Dynamic, Dynamic>;
    using dense_solver_t  = Eigen::PartialPivLU<matrix_t>;

    explicit MockSolver(const matrix_t& A) {
        n_locs_ = n_dofs_ = A.rows();
        invA_.A = &A;
        A_ = &A;
        Psi_.resize(n_locs_, n_dofs_); Psi_.setIdentity();
        W_.resize(n_locs_, n_locs_);   W_.setIdentity();
        D_ = vector_t::Ones(n_locs_).asDiagonal();
        lambda_saved_ = 1.0;
    }

    matrix_t apply_S(const matrix_t& M)  { return (*A_) * M; }
    matrix_t apply_St(const matrix_t& M) { return A_->transpose() * M; }
    const sparse_matrix_t& PsiNA() const { return Psi_; }

    // membri usati dagli stimatori
    const matrix_t* A_ = nullptr;
    std::optional<double> lambda_saved_;
    MockInvA invA_;
    std::optional<matrix_t> Ys_, Bs_, Us_;
    int n_dofs_ = 0, n_locs_ = 0, n_obs_ = 0, n_covs_ = 0;
    sparse_matrix_t Psi_;
    diag_matrix_t D_;
    std::vector<int> dirichlet_dofs_;
    std::vector<double> dirichlet_vals_;
    matrix_t X_;
    sparse_matrix_t W_;
    matrix_t U_, V_;
    matrix_t XtWX_;
    dense_solver_t invXtWX_;
    matrix_t invXtWXXtW_;

    // ================================================================================
    // INIZIO CODICE COPIATO VERBATIM DA fe_ls_elliptic (struct Hutch, Hutchpp, Xtrace, XNysTrace)
    // ================================================================================
    struct Hutch {
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

        int r_;
        int seed_;
    };
     
    struct Hutchpp {
        public:
        
        Hutchpp(int r = 100, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

        template <typename SolverType>
        double compute_S(SolverType& solver) const {
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            // Defining m as the closest multiple of 3 smaller than r
            int m = (r_ / 3) * 3; 
            int p = m / 3;
            //H, G creation and population with rademacher (H corresponds to matrix S in the Hutch++ paper)
            matrix_t H(solver.n_locs_, p);
            matrix_t G(solver.n_locs_, p);
            for (int i = 0; i < solver.n_locs_; ++i) {
                for (int j = 0; j < p; ++j) { 
                    H(i, j) = rademacher(rng);
                    G(i, j) = rademacher(rng);
                }
            }
            matrix_t SH = solver.apply_S(H);
            Eigen::ColPivHouseholderQR<matrix_t> qr(SH);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(solver.n_locs_, p); //I retain only the first p (m/3) columns of Q
            matrix_t SQ = solver.apply_S(Q);
            // Tr(Q'SQ) element-wise(Hadamard) product
            double tr_QSQ = (Q.cwiseProduct(SQ)).sum();
            // (I - QQ')G
            matrix_t G_p = G - Q * (Q.transpose() * G);
            matrix_t SG_p = solver.apply_S(G_p);
            // Tr(G_p'SG_p)
            double tr_GpSGp = (G_p.cwiseProduct(SG_p)).sum();
            return tr_QSQ + (tr_GpSGp / p);
        }

        template <typename SolverType>
        double compute_SS(SolverType& solver) const {
            
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            // Defining m as the closest multiple of 3 smaller than r
            int m = (r_ / 3) * 3; 
            int p = m / 3;
            //H, G creation and population with rademacher (H corresponds to matrix S in the Hutch++ paper)
            matrix_t H(solver.n_locs_, p);
            matrix_t G(solver.n_locs_, p);
            for (int i = 0; i < solver.n_locs_; ++i) {
                for (int j = 0; j < p; ++j) { 
                    H(i, j) = rademacher(rng);
                    G(i, j) = rademacher(rng);
                }
            }
            // S'SH
            matrix_t SSH = solver.apply_St(solver.apply_S(H)); //n_locs x p matrix, In the paper SSH = AS
            Eigen::ColPivHouseholderQR<matrix_t> qr(SSH);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(solver.n_locs_, p); //I retain only the first p (m/3) columns of Q
            // tr(Q'SSQ)
            matrix_t SQ = solver.apply_S(Q);
            // always Frobenius norm on S'S
            double tr_QSSQ = SQ.squaredNorm();
            // (I - QQ')G 
            matrix_t G_p = G - Q * (Q.transpose() * G);
            matrix_t SG_p = solver.apply_S(G_p);
            double tr_GpSSGp = SG_p.squaredNorm();
            return tr_QSSQ + (tr_GpSSGp / p); 
        }
        private:

        int r_;
        int seed_;
    };

    struct Xtrace {

        public:

        Xtrace(int r = 100, int seed = random_seed) noexcept : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {    }

        template <typename SolverType>
        double compute_S(SolverType& solver) const {
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            //Making sure m is an even integer (Xtrace requirement)
            int m = (r_ / 2) * 2; 
            int p = m / 2;
            matrix_t O(solver.n_locs_, p);
            matrix_t Y(solver.n_locs_, p);
            for (int i = 0; i < solver.n_locs_; ++i) {
                for (int j = 0; j < p; ++j) { 
                    O(i, j) = rademacher(rng);
                }
            }

            Y = solver.apply_S(O);
            Eigen::HouseholderQR<matrix_t> qr(Y);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(solver.n_locs_, p);
            matrix_t R = qr.matrixQR().topRows(p).triangularView<Eigen::Upper>();
            matrix_t Z = solver.apply_S(Q);
            matrix_t H = Q.transpose() * Z;
            matrix_t W = Q.transpose() * O;
            matrix_t T = Z.transpose() * O;
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
            return trace/p;
        }

        template <typename SolverType>
        double compute_SS(SolverType& solver) const {
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            //Making sure m is an even integer (Xtrace requirement)
            int m = (r_ / 2) * 2; 
            int p = m / 2;
            matrix_t O(solver.n_locs_, p);
            matrix_t Y(solver.n_locs_, p);
            for (int i = 0; i < solver.n_locs_; ++i) {
                for (int j = 0; j < p; ++j) { 
                    O(i, j) = rademacher(rng);
                }
            }

            Y = solver.apply_St(solver.apply_S(O));
            Eigen::HouseholderQR<matrix_t> qr(Y);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(solver.n_locs_, p);
            matrix_t R = qr.matrixQR().topRows(p).triangularView<Eigen::Upper>();
            matrix_t Z = solver.apply_St(solver.apply_S(Q));
            matrix_t H = Q.transpose() * Z;
            matrix_t W = Q.transpose() * O;
            matrix_t T = Z.transpose() * O;
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
            return trace/p;
        }

        private:

        int r_;
        int seed_;
    };


    struct XNysTrace {
    public:
        XNysTrace(int r = 100, int seed = random_seed) noexcept 
            : r_(r), seed_(seed == random_seed ? std::random_device()() : seed) {}

        template <typename SolverType>
        double compute_S(SolverType& solver) const {
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            
            int m = r_; 
            int N = solver.n_locs_;
            matrix_t O(N, m);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < m; ++j) {
                    O(i, j) = rademacher(rng);
                }
            }

        
            matrix_t Y = solver.apply_S(O);
            double nu = std::numeric_limits<double>::epsilon() * Y.norm() / std::sqrt(double(N));
            Y += nu * O;

            Eigen::HouseholderQR<matrix_t> qr(Y);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(N, m);
            matrix_t R = qr.matrixQR().topRows(m).triangularView<Eigen::Upper>();

            matrix_t H = O.transpose() * Y;
            H = 0.5 * (H + H.transpose());
            Eigen::LLT<matrix_t> llt(H);
            
            
            matrix_t C = llt.matrixL().transpose(); 
            matrix_t Cinv = C.triangularView<Eigen::Upper>().solve(matrix_t::Identity(m, m));
            matrix_t B = R * Cinv;

            
            Eigen::HouseholderQR<matrix_t> qrO(O);
            matrix_t QQ = qrO.householderQ() * matrix_t::Identity(N, m);
            matrix_t RR = qrO.matrixQR().topRows(m).triangularView<Eigen::Upper>();
            
            matrix_t WW = QQ.transpose() * O;
            matrix_t RRinvT = RR.triangularView<Eigen::Upper>().solve(matrix_t::Identity(m, m)).transpose();
            
            matrix_t SS = RRinvT;
            for(int i = 0; i < m; ++i) {
                SS.col(i).normalize(); 
            }

            std::vector<double> scale(m);
            for(int i = 0; i < m; ++i) {
                double norm_ww_i = WW.col(i).norm();
                double d_i = SS.col(i).dot(WW.col(i)); 
                scale[i] = (N - m + 1.0) / (N - norm_ww_i * norm_ww_i + std::pow(std::abs(d_i), 2));
            }

           
            matrix_t W = Q.transpose() * O;
            matrix_t Hinv = llt.solve(matrix_t::Identity(m, m)); 
            matrix_t BCinvT = B * Cinv.transpose();
            
            matrix_t S_mat(m, m);
            for(int i = 0; i < m; ++i) {
                S_mat.col(i) = BCinvT.col(i) * (1.0 / std::sqrt(Hinv(i, i)));
            }

            double normB2 = B.squaredNorm();
            double trace = 0.0;
            
            for(int i = 0; i < m; ++i) {
                double dSW_i = S_mat.col(i).dot(W.col(i));
                double normS2_i = S_mat.col(i).squaredNorm();
                
                double est_i = normB2 - normS2_i + std::pow(std::abs(dSW_i), 2) * scale[i] - nu * N;
                trace += est_i;
            }

            return trace / m;
        }

        template <typename SolverType>
        double compute_SS(SolverType& solver) const {
            fdapde_assert(solver.lambda_saved_.has_value());
            std::mt19937 rng(seed_);
            rademacher_distribution rademacher;
            
            int m = r_; 
            int N = solver.n_locs_;
            matrix_t O(N, m);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < m; ++j) {
                    O(i, j) = rademacher(rng);
                }
            }

            
            matrix_t Y = solver.apply_St(solver.apply_S(O));
            double nu = std::numeric_limits<double>::epsilon() * Y.norm() / std::sqrt(double(N));
            Y += nu * O;

            Eigen::HouseholderQR<matrix_t> qr(Y);
            matrix_t Q = qr.householderQ() * matrix_t::Identity(N, m);
            matrix_t R = qr.matrixQR().topRows(m).triangularView<Eigen::Upper>();

            matrix_t H = O.transpose() * Y;
            H = 0.5 * (H + H.transpose());
            Eigen::LLT<matrix_t> llt(H);
            matrix_t C = llt.matrixL().transpose();
            matrix_t Cinv = C.triangularView<Eigen::Upper>().solve(matrix_t::Identity(m, m));
            matrix_t B = R * Cinv;

            Eigen::HouseholderQR<matrix_t> qrO(O);
            matrix_t QQ = qrO.householderQ() * matrix_t::Identity(N, m);
            matrix_t RR = qrO.matrixQR().topRows(m).triangularView<Eigen::Upper>();
            
            matrix_t WW = QQ.transpose() * O;
            matrix_t RRinvT = RR.triangularView<Eigen::Upper>().solve(matrix_t::Identity(m, m)).transpose();
            matrix_t SS = RRinvT;
            for(int i = 0; i < m; ++i) {
                SS.col(i).normalize();
            }

            std::vector<double> scale(m);
            for(int i = 0; i < m; ++i) {
                double norm_ww_i = WW.col(i).norm();
                double d_i = SS.col(i).dot(WW.col(i)); 
                scale[i] = (N - m + 1.0) / (N - norm_ww_i * norm_ww_i + std::pow(std::abs(d_i), 2));
            }

            matrix_t W = Q.transpose() * O;
            matrix_t Hinv = llt.solve(matrix_t::Identity(m, m));
            matrix_t BCinvT = B * Cinv.transpose();
            
            matrix_t S_mat(m, m);
            for(int i = 0; i < m; ++i) {
                S_mat.col(i) = BCinvT.col(i) * (1.0 / std::sqrt(Hinv(i, i)));
            }

            double normB2 = B.squaredNorm();
            double trace = 0.0;
            
            for(int i = 0; i < m; ++i) {
                double dSW_i = S_mat.col(i).dot(W.col(i));
                double normS2_i = S_mat.col(i).squaredNorm();
                double est_i = normB2 - normS2_i + std::pow(std::abs(dSW_i), 2) * scale[i] - nu * N;
                trace += est_i;
            }

            return trace / m;
        }

    private:
        int r_;
        int seed_;
    };
    // ================================================================================
    // FINE CODICE COPIATO VERBATIM
    // ================================================================================
};

// ------------------------------------------------------------------------------------
// Generazione delle matrici di test (stesso esperimento del paper)
// ------------------------------------------------------------------------------------
static MatrixXd haar_orthogonal(int N, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    MatrixXd G(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) G(i, j) = g(rng);
    HouseholderQR<MatrixXd> qr(G);
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(N, N);
    VectorXd d = qr.matrixQR().diagonal();
    for (int j = 0; j < N; ++j)
        if (d(j) < 0) Q.col(j) *= -1.0;
    return Q;
}

static VectorXd make_spectrum(const std::string& type, int N) {
    VectorXd lam(N);
    if (type == "flat") {
        for (int i = 0; i < N; ++i) lam(i) = 3.0 - 2.0 * i / (N - 1.0);
    } else if (type == "poly") {
        for (int i = 1; i <= N; ++i) lam(i - 1) = std::pow(double(i), -2.0);
    } else if (type == "exp") {
        for (int i = 0; i < N; ++i) lam(i) = std::pow(0.7, double(i));
    } else {   // step
        for (int i = 0; i < N; ++i) lam(i) = (i < 50) ? 1.0 : 1e-3;
    }
    return lam;
}

// ------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int n_trials = (argc > 1) ? std::atoi(argv[1]) : 100;   // il paper usa 1000
    const int N        = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int base_seed = 159875;

    std::vector<int> ms;
    for (int m = 20; m <= 300; m += 20) ms.push_back(m);

    std::vector<std::string> types   = {"flat", "poly", "exp", "step"};
    std::vector<std::string> methods = {"Hutch", "Hutch++", "XTrace", "XNysTrace"};

    std::vector<std::vector<std::vector<double>>> sum_err(
        types.size(), std::vector<std::vector<double>>(methods.size(), std::vector<double>(ms.size(), 0.0)));
    auto sum_time = sum_err;

    std::ofstream raw("benchmark_trace_raw.csv");
    raw << "MatrixType,Method,m,Trial,Time_ms,Estimate,True_Value,RelError\n";
    raw.precision(17);

    std::cout << "\n=== BENCHMARK STIMATORI DI TRACCIA (N=" << N << ", trials=" << n_trials << ") ===\n";

    for (int t = 0; t < n_trials; ++t) {
        std::mt19937 rng_U(base_seed + 1000003 * t);
        MatrixXd U = haar_orthogonal(N, rng_U);

        for (size_t ti = 0; ti < types.size(); ++ti) {
            VectorXd lam = make_spectrum(types[ti], N);
            MatrixXd A = U * lam.asDiagonal() * U.transpose();
            A = 0.5 * (A + A.transpose());
            double true_tr = lam.sum();

            for (size_t mi = 0; mi < methods.size(); ++mi) {
                for (size_t k = 0; k < ms.size(); ++k) {
                    int m = ms[k];
                    int seed = base_seed + 7919 * t + 104729 * (int)mi + 31 * m;

                    // solver "fresco" ad ogni chiamata: Hutch::compute_S mette in cache Us_/Ys_/Bs_ nel solver
                    MockSolver solver(A);
                    double est = 0.0;

                    auto start = high_resolution_clock::now();
                    if (mi == 0)      { MockSolver::Hutch     e(m, seed); est = e.compute_S(solver); }
                    else if (mi == 1) { MockSolver::Hutchpp   e(m, seed); est = e.compute_S(solver); }
                    else if (mi == 2) { MockSolver::Xtrace    e(m, seed); est = e.compute_S(solver); }
                    else              { MockSolver::XNysTrace e(m, seed); est = e.compute_S(solver); }
                    auto stop = high_resolution_clock::now();

                    double time_ms = duration<double, std::milli>(stop - start).count();
                    double err = std::abs(est - true_tr) / true_tr;

                    sum_err[ti][mi][k]  += err;
                    sum_time[ti][mi][k] += time_ms;
                    raw << types[ti] << "," << methods[mi] << "," << m << "," << t << ","
                        << time_ms << "," << est << "," << true_tr << "," << err << "\n";
                }
            }
        }
        std::cout << "Trial " << (t + 1) << "/" << n_trials << " completato\r" << std::flush;
    }
    raw.close();

    std::ofstream summ("benchmark_trace_summary.csv");
    summ.precision(17);
    summ << "MatrixType,Method,m,MeanRelError,MeanTime_ms,NTrials\n";
    for (size_t ti = 0; ti < types.size(); ++ti)
        for (size_t mi = 0; mi < methods.size(); ++mi)
            for (size_t k = 0; k < ms.size(); ++k)
                summ << types[ti] << "," << methods[mi] << "," << ms[k] << ","
                     << sum_err[ti][mi][k] / n_trials << "," << sum_time[ti][mi][k] / n_trials << ","
                     << n_trials << "\n";
    summ.close();

    std::cout << "\nBenchmark completato! File 'benchmark_trace_raw.csv' e 'benchmark_trace_summary.csv' generati.\n";
    return 0;
}