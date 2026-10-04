#include <fdaPDE/models.h>

#include "fe_ls_elliptic.h"
#include "sr.h"

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace fdapde;

// Funzione RMSE
template <typename T, typename S> double rmse(const T& lhs, const S& rhs) {
    fdapde_assert(lhs.rows() == rhs.rows());
    int n = lhs.rows();
    double sse = 0;
    for (int i = 0; i < n; ++i) { 
        sse += std::pow(lhs(i, 0) - rhs(i), 2); 
    }
    return std::sqrt((1. / n) * sse);
}

int main() {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;

    int seed = 159875;   
    int r_samples = 150; 

    std::cout << "=========================================================\n";
    std::cout << " TEST fdaPDE: Risolto il mistero di Tr(S'S) per XTrace!  \n";
    std::cout << "=========================================================\n\n";

    // ------------------------ geometry
    std::cout << "[1] Generazione Mesh e Dati (UnitSquare)..." << std::endl;
    Triangulation<2, 2> D = Triangulation<2, 2>::UnitSquare(30);

    // ------------------------ generate data
    auto generate_data = [](const matrix_t& locs) -> matrix_t {
        int n_locs = locs.rows();
        matrix_t data(n_locs, 1);
        for (int i = 0; i < n_locs; ++i) {
            double p_x = locs(i, 0);
            double p_y = locs(i, 1);
            data(i, 0) = std::sin(1 * p_x) * std::sin(3 * p_y) + std::sin(5 * p_x) * std::sin(7 * p_y);
        }
        return data;
    };
    
    Triangulation<2, 2> data_grid = Triangulation<2, 2>::UnitSquare(40);
    matrix_t y_clean = generate_data(data_grid.nodes());

    std::vector<double> y;
    double min = y_clean.minCoeff();
    double max = y_clean.maxCoeff();
    std::mt19937 gen(seed);
    std::normal_distribution obs_noise(0.0, 0.05 * std::abs(max - min));
    for (int ctr = 0; ctr < y_clean.rows(); ++ctr) { y.push_back(y_clean(ctr, 0) + obs_noise(gen)); }

    GeoFrame data(D);
    auto& layer = data.insert_scalar_layer<POINT>("layer", data_grid.nodes());
    layer.load_vec("y", y);

    // ------------------------ physics
    std::cout << "[2] Assemblaggio Forme Bilineari/Lineari (Laplaciano)..." << std::endl;
    FeSpace Vh(D, P1<1>);
    int n_dofs = Vh.n_dofs();
    TrialFunction f(Vh);
    TestFunction  v(Vh);

    auto a = integral(D)(dot(grad(f), grad(v)));
    ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 0.0; })> force;
    auto F = integral(D)(force * v);

    // ------------------------ modeling
    SRPDE model("y ~ f", data, fe_ls_elliptic(a, F));
    std::cout << "[3] Modello SRPDE inizializzato. Dofs: " << n_dofs << " | Obs: " << data[0].rows() << "\n\n";

    // ------------------------ TEST STIMATORI
    std::cout << "--- Confronto Stimatori (Lambda = 1e-4, r = " << r_samples << ") ---\n";
    double pilot_lambda = 1e-4;
    model.fit(pilot_lambda); 

    // Istanziazione
    fdapde::internals::fe_ls_elliptic::Hutch   hutch_est(r_samples, seed);
    fdapde::internals::fe_ls_elliptic::Hutchpp hutchpp_est(r_samples, seed);
    fdapde::internals::fe_ls_elliptic::Xtrace  xtrace_est(r_samples, seed);

    using namespace std::chrono;
    auto run_test = [&](const std::string& name, auto& estimator) {
        auto start = high_resolution_clock::now();
        double trS  = model.edf(estimator);
        double trSS = model.edf_SS(estimator);
        auto stop = high_resolution_clock::now();
        double time = duration<double, std::milli>(stop - start).count();
        
        std::cout << std::left << std::setw(15) << name 
                  << std::setw(15) << trS 
                  << std::setw(15) << trSS 
                  << std::setw(15) << time << "\n";
    };

    std::cout << std::left << std::setw(15) << "Stimatore" 
              << std::setw(15) << "Tr(S)" 
              << std::setw(15) << "Tr(S'S)" 
              << std::setw(15) << "Tempo (ms)" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    // Decommenta se non hai messo ExactTrace in fe_ls_elliptic.h
    run_test("Hutchinson", hutch_est);
    run_test("Hutch++", hutchpp_est);
    run_test("XTrace", xtrace_est);
    
    std::cout << "\n";

    return 0;
}