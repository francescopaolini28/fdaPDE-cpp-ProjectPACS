#include <fdaPDE/models.h>  // include gran parte della libreria fdaPDE



// questi sono i vosti header, potete includere tutti gli header che volete
#include "fe_ls_elliptic.h"
#include "sr.h"

#include <Eigen/Dense>


// ....

using namespace fdapde;

// una semplice utlità per valutare l'RMSE
template <typename T, typename S> double rmse(const T& lhs, const S& rhs) {
    fdapde_assert(lhs.size() == rhs.size());
    int n = lhs.size();
    double sse = 0;
    for (int i = 0; i < n; ++i) { sse += std::pow(lhs[i] - rhs[i], 2); }
    return std::sqrt((1. / n) * sse);
}

int main() {
    using matrix_t = Eigen::Matrix<double, Dynamic, Dynamic>;
    using vector_t = Eigen::Matrix<double, Dynamic, 1>;

    int seed = 159875;   // script-wise seed, repoducibility

    // ------------------------ geometry
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

    //std::cout << layer << std::endl;

    // ------------------------ physics
    FeSpace Vh(D, P1<1>);
    int n_dofs = Vh.n_dofs();
    TrialFunction f(Vh);
    TestFunction  v(Vh);

    
    auto a = integral(D)(dot(grad(f), grad(v)));
    ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 0.0; })> force;
    auto F = integral(D)(force * v);

    // ------------------------ modeling
    SRPDE model("y ~ f", data, fe_ls_elliptic(a, F));

    // ------------------------ calibration
    auto gcv = model.gcv(100, seed);   // index to optimize
    GridSearch<1> optimizer;                  // optimizer

    std::vector<double> lambda_grid(13);
    for (int i = 0; i < 13; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
    optimizer.optimize(gcv, lambda_grid);

    // final fit with optimal smoothing parameter
    model.fit(optimizer.optimum());

    // ------------------------ postprocess
    std::vector<double> gcv_values, gcv_optimum;    
    for (auto v : optimizer.values())  { gcv_values .push_back(v); }
    for (auto v : optimizer.optimum()) { gcv_optimum.push_back(v); }
    write_csv("gcv_values.csv" , gcv_values    );
    write_csv("gcv_optimum.csv", gcv_optimum   );
    write_csv("f.csv"          , model.f()     );
    write_csv("fitted.csv"     , model.fitted());
    
    return 0;
}
