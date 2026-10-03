#include <fdaPDE/models.h>  // include gran parte della libreria fdaPDE

// questi sono i vostri header (assicurati che il nome del file coincida con il tuo locale)
#include "fe_ls_elliptic.h"
#include "sr.h" // Se l'hai salvato come spatial_regression.h, cambia l'include

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace fdapde;

// una semplice utlità per valutare l'RMSE
// una semplice utilità per valutare l'RMSE
template <typename T, typename S> double rmse(const T& lhs, const S& rhs) {
    fdapde_assert(lhs.rows() == rhs.rows());
    int n = lhs.rows();
    double sse = 0;
    for (int i = 0; i < n; ++i) { 
        // lhs è MatrixXd -> lhs(i, 0)
        // rhs è VectorXd -> rhs(i)
        sse += std::pow(lhs(i, 0) - rhs(i), 2); 
    }
    return std::sqrt((1. / n) * sse);
}

int main() {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;

    int seed = 159875;   // script-wise seed, reproducibility
    int r_samples = 150; // Numero di campioni di Rademacher per le tracce

    std::cout << "====================================================\n";
    std::cout << "   TEST fdaPDE: SRPDE con Stimatori Hutch/Hutch++   \n";
    std::cout << "====================================================\n\n";

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

            double term1 = std::sin(1 * p_x) * std::sin(3 * p_y);
            double term2 = std::sin(5 * p_x) * std::sin(7 * p_y);
            data(i, 0) = term1 + term2;
        }
        return data;
    };
    
    Triangulation<2, 2> data_grid = Triangulation<2, 2>::UnitSquare(40);
    // regular grid
    matrix_t y_clean = generate_data(data_grid.nodes());
    write_csv("y_clean.csv", y_clean);

    // generate noisy data, adding gaussian distributed noise
    std::vector<double> y;
    double min = y_clean.minCoeff();
    double max = y_clean.maxCoeff();
    // add noise
    std::mt19937 gen(seed);
    std::normal_distribution obs_noise(0.0, 0.05 * std::abs(max - min));
    for (int ctr = 0; ctr < y_clean.rows(); ++ctr) { y.push_back(y_clean(ctr, 0) + obs_noise(gen)); }
    write_csv("y_noise.csv", y);

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

    // ------------------------ TEST MANUALE STIMATORI
    std::cout << "--- Test invocazione manuale Strategy Pattern ---\n";
    double pilot_lambda = 1e-4;
    model.fit(pilot_lambda); 

    // Istanziamo gli stimatori passandogli numero di iterazioni e seed
    fdapde::internals::fe_ls_elliptic::Hutch   hutch_est(r_samples, seed);
    fdapde::internals::fe_ls_elliptic::Hutchpp hutchpp_est(r_samples, seed);

    auto start = std::chrono::high_resolution_clock::now();
    double trS = model.edf(hutch_est);       // Chiama internamente Hutch
    double trSS = model.edf_SS(hutchpp_est); // Chiama internamente Hutch++
    auto stop = std::chrono::high_resolution_clock::now();
    
    std::cout << "    Lambda Pilota : " << pilot_lambda << "\n";
    std::cout << "    Tr(S)   [Hutch]   : " << trS << "\n";
    std::cout << "    Tr(S'S) [Hutch++] : " << trSS << "\n";
    std::cout << "    Tempo calcolo     : " << std::chrono::duration<double, std::milli>(stop - start).count() << " ms\n\n";

    // ------------------------ calibration
    std::cout << "[4] Avvio GridSearch per calibrazione parametro di smoothing...\n";
    
    // Uso PSE (Predictive Squared Error) perché è il criterio che forza il codice 
    // a calcolare SIA Tr(S) con Hutch SIA Tr(S'S) con Hutch++ in automatico!
    auto pse_criterion = model.pse(r_samples, seed);   
    GridSearch<1> optimizer;                  

    std::vector<double> lambda_grid(13);
    for (int i = 0; i < 13; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
    
    start = std::chrono::high_resolution_clock::now();
    optimizer.optimize(pse_criterion, lambda_grid);
    stop = std::chrono::high_resolution_clock::now();

    std::cout << "    GridSearch completata in " << std::chrono::duration<double>(stop - start).count() << " secondi.\n";
    std::cout << "    Ottimo trovato (Lambda) : " << std::scientific << optimizer.optimum()[0] << "\n\n";

    // final fit with optimal smoothing parameter
    model.fit(optimizer.optimum());

    // ------------------------ postprocess
    std::cout << "[5] Esportazione CSV e calcolo Errore...\n";
    
    double error = rmse(y_clean, model.fitted());
    std::cout << "    => RMSE finale rispetto ai dati clean: " << std::fixed << std::setprecision(5) << error << "\n\n";

    std::vector<double> criterion_values, criterion_optimum;    
    for (auto v : optimizer.values())  { criterion_values.push_back(v); }
    for (auto v : optimizer.optimum()) { criterion_optimum.push_back(v); }
    
    write_csv("pse_values.csv" , criterion_values  );
    write_csv("pse_optimum.csv", criterion_optimum );
    write_csv("f.csv"          , model.f()         );
    write_csv("fitted.csv"     , model.fitted()    );
    
    std::cout << "Run terminata con successo!" << std::endl;
    return 0;
}