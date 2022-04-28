#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <math.h>
#include "macros.h"
#include <vector>

struct workspace{
    Eigen::MatrixXd P;
    Eigen::MatrixXd A;
    Eigen::MatrixXd At;
    Eigen::MatrixXd KKT;
    Eigen::MatrixXd KKT_tmp;
    Eigen::VectorXd rho_vec;
    Eigen::MatrixXd rho_inv_mat;
    Eigen::VectorXd rhs;
    Eigen::VectorXd rhs_prev;
    Eigen::VectorXd x, y, z;
    Eigen::VectorXd x_tilde, z_tilde;
    Eigen::VectorXd x_delta, y_delta, z_delta;
    Eigen::VectorXd q, upper, lower;
    double alpha;
    double sigma;
    double rho;
    double rho_estimate;
    double adapt_rho_tol;
    double eps_pri_limit, eps_dual_limit, eps_pinf_limit, eps_dinf_limit;
    int m;
    int n;
    bool compute_objective;
    double obj_val, pri_res, dua_res;
};

Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic> matrixGenerate(const int m, const int n){
    Eigen::MatrixXd A(m, n);
    A.setZero();
    return A;
}

void vec_init(Eigen::VectorXd &vec, int size){
    vec = Eigen::VectorXd(size);
    vec.setZero();
}

void kktMerge(Eigen::MatrixXd& KKT, const Eigen::MatrixXd& P, const Eigen::MatrixXd& A, const Eigen::MatrixXd &rho){
    KKT<<P, A.transpose(), A, rho;
}

double _infty_norm(Eigen::VectorXd vec, int size){
    double res = -99999999999999;
    for(int i = 0; i<size; i++){
        res = std::max(res, vec[i]);
    }
    return std::abs(res);
}

bool _isConvergent(workspace &workspace){
    double eps_pri = _infty_norm((workspace.A * workspace.x - workspace.z), workspace.m);
    double eps_dual = _infty_norm((workspace.P * workspace.x + workspace.q+workspace.At*workspace.y), workspace.n);
//    printf("primal residual is %f, dual residual is %f\n", eps_pri, eps_dual);
    return eps_pri<workspace.eps_pri_limit and eps_dual<workspace.eps_dual_limit;
}

double compute_rho_estimate(workspace &workspace){
    int n = workspace.n, m = workspace.m;
    double pri_res, dua_res;
    double tol = 1e-10;
    double rho_estimate;

    pri_res = _infty_norm(workspace.rhs_prev.tail(m), m);
    dua_res = _infty_norm(workspace.rhs_prev.head(n), n);

    double eps_pri = _infty_norm((workspace.A * workspace.x - workspace.z), workspace.m);
    double eps_dual = _infty_norm((workspace.P * workspace.x + workspace.q+workspace.At*workspace.y), workspace.n);

    pri_res /= (eps_pri+tol);
    dua_res /=(eps_dual+tol);

    rho_estimate = workspace.rho * std::sqrt(pri_res / dua_res);
    rho_estimate = std::min(std::max(rho_estimate, 1e-06), 1e06);
    return rho_estimate;
}

int update_rho(workspace& workspace, double rho_new){
    int exit = 0;
    int m = workspace.m, n = workspace.n;
    if(rho_new <=0) return 1;

    workspace.rho = std::min(std::max(rho_new, 1e-6),1e6);

    //update rho_vec and KKT matrix
    for(int i =0; i<m;i++){
        workspace.rho_vec[i] = workspace.rho;
        workspace.KKT(i+n, i+n) = -1.0 / workspace.rho;
    }
    return exit;
}
int adapt_rho(workspace &workspace){
    int flag = 0;
    double rho_new;

    rho_new = compute_rho_estimate(workspace);
    workspace.rho_estimate = rho_new;

    if((rho_new > workspace.rho * workspace.adapt_rho_tol) ||
        (rho_new < workspace.rho /  workspace.adapt_rho_tol)){
            update_rho(workspace, rho_new);
    }
    return flag;
}
double compute_obj_val(workspace& workspace){
    auto obj_val = 0.5 * (workspace.x.transpose() * workspace.P * workspace.x) + workspace.q.transpose() * workspace.x;
    return obj_val[0];
}



void compute_rhs(workspace &workspace){
    int i;
    int m = workspace.m, n = workspace.n;
    for(i = 0; i < n; i++){ //x
        workspace.rhs[i] = workspace.sigma * workspace.rhs_prev[i] - workspace.q[i];
    }
    for(i = 0; i<m; i++){ //z
        workspace.rhs[i + n] = workspace.rhs_prev[i + n] - (1.0/workspace.rho_vec[i]) * workspace.y[i];
    }

//    std::cout<<"y:\n"<<workspace.y<<std::endl;
//    std::cout<<"before solve:\n"<<workspace.rhs<<std::endl;
}

void update_xz_tilde(workspace &workspace){
    int m = workspace.m, n = workspace.n;
    compute_rhs(workspace);
    workspace.rhs = workspace.KKT_tmp.colPivHouseholderQr().solve(workspace.rhs);
//    std::cout<<workspace.rhs.head(n)<<std::endl;
    workspace.rhs.tail(m) = workspace.A * workspace.rhs.head(n); //z_tilde
//    std::cout<<"solve:\n"<<workspace.rhs<<std::endl;
//    std::cout<<std::endl;
}

void update_x(workspace& workspace){
    int n = workspace.n;
    int i = 0;
    for(i = 0; i<n; i++){
        workspace.x[i] = workspace.alpha * workspace.rhs[i] + (1.0-workspace.alpha) * workspace.rhs_prev[i];
    }
//    std::cout<<"x:"<<workspace.x<<std::endl;

    for(i = 0; i<n; i++){
        workspace.x_delta[i] = workspace.x[i] - workspace.rhs_prev[i];
    }
//    std::cout<<"x_delta:"<<workspace.x_delta<<std::endl;
}
void project(workspace &workspace){
    int i, m = workspace.m;
    for(i = 0; i<m;i++){
        workspace.z[i] = std::min(std::max(workspace.z[i], workspace.lower[i]), workspace.upper[i]);
    }
}
void update_z(workspace& workspace){
    int m = workspace.m, n = workspace.n;
    for(int i = 0; i<m;i++){
        workspace.z[i] = workspace.alpha * workspace.rhs[i + n] \
        + (1.0-workspace.alpha) * workspace.rhs_prev[i+n] \
        + (1.0/workspace.rho_vec[i]) * workspace.y[i];
//        std::cout<<"z["<<i<<"]: "<<workspace.z[i]<<std::endl;
//        std::cout<<"rho: "<<1.0/workspace.rho_vec[i]<<std::endl;
    }
    project(workspace);
//    std::cout<<"z:\n"<<workspace.z<<std::endl;
}

void update_y(workspace& workspace){
    int m = workspace.m;
    int n = workspace.n;
    for(int i =0; i<m;i++){
        workspace.y_delta[i] = workspace.rho_vec[i] * (\
        workspace.alpha * workspace.rhs[i+n] \
        + (1.0 - workspace.alpha) * workspace.rhs_prev[i+n] - workspace.z[i]);

        workspace.y[i] += workspace.y_delta[i];
//        std::cout<<workspace.y_delta[i]<<" "<<workspace.y[i]<<std::endl;
    }
}
double compute_pri_res(workspace &workspace){
    int m = workspace.m;
    workspace.rhs_prev.tail(m) = workspace.A * workspace.x - workspace.z;
    //Scaling in the future, maybe?
    return _infty_norm(workspace.rhs_prev.tail(m), m);
}

double compute_dua_res(workspace &workspace){
    //NB: Use x_prev as temporaty vector
    int n = workspace.n;
    int m = workspace.m;
    workspace.rhs_prev.head(n) = workspace.q;
    workspace.rhs_prev.head(n) = workspace.P * workspace.x + workspace.rhs_prev.head(n);

    if(m > 0){
        workspace.rhs_prev.head(n) = workspace.At * workspace.y + workspace.rhs_prev.head(n);
    }
    return _infty_norm(workspace.rhs_prev.head(n), n);

}
void update_info(workspace &workspace){
    int m = workspace.m, n = workspace.n;
    if(workspace.compute_objective){
        workspace.obj_val = compute_obj_val(workspace);
    }
    if(m==0){
        workspace.pri_res = 0.0;
    }else{
        workspace.pri_res = compute_pri_res(workspace);
    }

    workspace.dua_res = compute_dua_res(workspace);

}

bool _PrimalInfeasibility(workspace& workspace){
    int n = workspace.n;
    if(_infty_norm(workspace.y_delta, n) < workspace.eps_pinf_limit){
        return false;
    }
    double cond1 = _infty_norm(workspace.At*workspace.y_delta, n);
//    double cond2 =
    }

bool admm(workspace & workspace){
    int n = workspace.n;
    int m = workspace.m;
    int iter = 0;
    bool flag = false;
    while(!flag and iter<MAX_ITER){
        workspace.KKT_tmp = workspace.KKT;
        std::swap(workspace.rhs_prev, workspace.rhs);
        update_xz_tilde(workspace);// fine
        update_x(workspace);
        update_z(workspace);
        update_y(workspace);
        workspace.rhs.head(n) = workspace.x;
        workspace.rhs.tail(m) = workspace.z;

        update_info(workspace);
//        std::cout<<"Iter:"<<iter++<<std::endl;
//        std::cout<<"xz_prev:\n"<<workspace.rhs_prev<<std::endl;
        flag = _isConvergent(workspace);
        iter++;
    }
    if(!flag){
        printf("The problem is cannot be solved.");
        return !flag;
    }
    if(iter == MAX_ITER and flag==false){
        printf("The problem cannot be solved by now");
        return flag;
    }
    printf("ADMM ENDS AFTER %d ITERATIONS\n", iter);
    return flag;
}

void polish(workspace &workspace){
    int m = workspace.m, n = workspace.n;
    for(int i = 0; i<5; i++){
        std::swap(workspace.rhs_prev, workspace.rhs);
//        std::cout<<"rhs_prev:\n"<<workspace.rhs_prev<<std::endl;
//        std::cout<<"rhs:\n"<<workspace.rhs<<std::endl;
        update_xz_tilde(workspace);// fine
        update_x(workspace);
        update_z(workspace);
        update_y(workspace);
        workspace.rhs.head(n) = workspace.x;
        workspace.rhs.tail(m) = workspace.z;
        update_info(workspace);
    }
}
void init_data_default(workspace& workspace, int m, int n){
    int size = m + n;
    workspace.m = m;
    workspace.n = n;
    workspace.alpha = ALPHA;
    workspace.P = matrixGenerate(n, n);
    workspace.A = matrixGenerate(m, n);
    workspace.At = matrixGenerate(n, m);
    workspace.rho = RHO;
    workspace.rho_inv_mat = (-1 / workspace.rho) * Eigen::MatrixXd::Identity(m,m);
    workspace.KKT = matrixGenerate(size, size);
    workspace.adapt_rho_tol = 5.0;
    workspace.rho_estimate = 0.1;
    workspace.sigma = 1e-6;
    vec_init(workspace.rho_vec, m);
    for(int i = 0; i<m;i++){
        workspace.rho_vec[i] = workspace.rho;
    }

    vec_init(workspace.rhs, size);
    vec_init(workspace.rhs_prev, size);
    vec_init(workspace.q, n);
    vec_init(workspace.x, n);
    vec_init(workspace.x_delta, n);
    vec_init(workspace.z, m);
    vec_init(workspace.y, m);
    vec_init(workspace.y_delta,m);
    vec_init(workspace.z_delta, m);
    workspace.eps_dual_limit = (double)MAX_TOL;
    workspace.eps_pri_limit = (double)MAX_TOL;
    workspace.eps_pinf_limit = (double)MAX_TOL;
    workspace.eps_dinf_limit = (double)MAX_TOL;
    workspace.compute_objective = COMPUTE_OBJ;
    workspace.obj_val = -1.0;
    workspace.pri_res = 0.0;
    workspace.dua_res = 0.0;
    vec_init(workspace.upper, m);
    vec_init(workspace.lower, m);
}

//void update_matrix(workspace& workspace, std::vector<double> &P, std::vector<double> &A,
//                   std::vector<double> &q, std::vector<double> &upper, std::vector<double> &lower){
//    workspace.P = Eigen::Map<Eigen::MatrixXd ,Eigen::Unaligned>(P.data(), P.size());
//    workspace.A = Eigen::Map<Eigen::MatrixXd ,Eigen::Unaligned>(A.data(), A.size());
//    workspace.At = workspace.A.transpose();
//    workspace.q = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(q.data(), q.size());
//    workspace.lower = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(lower.data(), lower.size());
//    workspace.upper = Eigen::Map<Eigen::VectorXd,Eigen::Unaligned>(upper.data(), upper.size());
//    workspace.KKT<<workspace.P,workspace.At, workspace.A,workspace.rho_inv_mat;
//    workspace.KKT_tmp = workspace.KKT;
//}
int main() {
    int m=70, n = 30;
//    int m = 2, n = 3
    int size = m + n;
    workspace workspace;
    init_data_default(workspace,m ,n);

//        workspace.P<<1.0000, 0.353553, 0.353553, 1.00000;
//        workspace.A<<0.706628, 0.999323, 0.998647, 0.0, 0.0,0.999323;
//        workspace.q<<0.5, 0.707106769;
//        workspace.upper<<1.413256, 1.398106, 0.989279;
//        workspace.lower<<1.413256, 0.0, 0.0;
    workspace.P<<0.253,0.871,0.494,0.576,0.365,0.599,0.446,0.585,0.833,0.406,0.553,0.191,0.978,0.020,0.757,0.009,0.757,0.241,0.107,0.765,0.672,0.065,0.525,0.811,0.170,0.523,0.891,0.636,0.713,0.235,0.109,0.234,0.638,0.939,0.053,0.371,0.044,0.400,0.642,0.338,0.084,0.988,0.055,0.551,0.373,0.354,0.837,0.750,0.200,0.830,0.645,0.308,0.586,0.551,0.586,0.543,0.241,0.832,0.362,0.956,0.337,0.158,0.469,0.145,0.224,0.804,0.907,0.394,0.150,0.381,0.711,0.192,0.226,0.777,0.628,0.066,0.018,0.350,0.895,0.166,0.979,0.525,0.716,0.408,0.662,0.459,0.962,0.099,0.939,0.734,0.606,0.780,0.016,0.711,0.065,0.168,0.112,0.647,0.527,0.538,0.348,0.038,0.020,0.679,0.558,0.516,0.255,0.527,0.339,0.572,0.000,0.844,0.217,0.449,0.034,0.178,0.999,0.935,0.041,0.020,0.952,0.024,0.207,0.203,0.909,0.069,0.651,0.527,0.609,0.854,0.373,0.511,0.317,0.645,0.195,0.301,0.178,0.856,0.743,0.724,0.537,0.042,0.296,0.514,0.064,0.931,0.088,0.500,0.336,0.504,0.593,0.860,0.960,0.796,0.232,0.852,0.126,0.578,0.888,0.203,0.271,0.203,0.096,0.236,0.562,0.702,0.992,0.515,0.070,0.812,0.614,0.287,0.356,0.026,0.764,0.406,0.947,0.886,0.862,0.291,0.342,0.353,0.857,0.896,0.651,0.136,0.798,0.604,0.214,0.527,0.923,0.785,0.510,0.429,0.212,0.115,0.502,0.111,0.339,0.125,0.000,0.679,0.984,0.887,0.306,0.302,0.719,0.100,0.191,0.789,0.302,0.180,0.483,0.245,0.881,0.452,0.281,0.053,0.915,0.373,0.101,0.748,0.817,0.750,0.050,0.158,0.512,0.481,0.389,0.876,0.954,0.840,0.558,0.255,0.005,0.385,0.050,0.247,0.082,0.103,0.908,0.190,0.110,0.120,0.278,0.423,0.896,0.390,0.580,0.276,0.675,0.217,0.924,0.448,0.109,0.665,0.184,0.435,0.903,0.244,0.156,0.711,0.492,0.920,0.430,0.411,0.204,0.810,0.397,0.928,0.626,0.372,0.335,0.614,0.162,0.713,0.131,0.470,0.179,0.837,0.573,0.842,0.858,0.898,0.413,0.240,0.191,0.422,0.371,0.639,0.481,0.183,0.044,0.852,0.355,0.889,0.577,0.587,0.587,0.852,0.768,0.172,0.839,0.609,0.365,0.333,0.648,0.376,0.681,0.597,0.005,0.572,0.344,0.419,0.879,0.339,0.068,0.832,0.902,0.680,0.133,0.745,0.168,0.182,0.668,0.797,0.921,0.579,0.993,0.354,0.984,0.930,0.521,0.336,0.677,0.350,0.967,0.767,0.908,0.838,0.673,0.069,0.690,0.801,0.313,0.865,0.035,0.794,0.150,0.759,0.880,0.025,0.186,0.367,0.889,0.769,0.526,0.899,0.960,0.898,0.602,0.746,0.592,0.738,0.725,0.063,0.886,0.176,0.942,0.470,0.396,0.257,0.007,0.252,0.009,0.559,0.395,0.949,0.016,0.703,0.923,0.799,0.451,0.526,0.957,0.162,0.594,0.913,0.994,0.129,0.265,0.713,0.709,0.856,0.578,0.761,0.211,0.980,0.232,0.550,0.110,0.511,0.046,0.826,0.417,0.578,0.719,0.280,0.299,0.593,0.705,0.985,0.440,0.589,0.404,0.836,0.190,0.501,0.820,0.805,0.485,0.951,0.677,0.992,0.100,0.901,0.466,0.010,0.183,0.877,0.242,0.040,0.357,0.193,0.337,0.357,0.121,0.011,0.737,0.106,0.534,0.766,0.835,0.620,0.844,0.122,0.960,0.403,0.263,0.690,0.659,0.591,0.100,0.216,0.504,0.405,0.435,0.208,0.117,0.482,0.973,0.370,0.995,0.900,0.698,0.903,0.438,0.772,0.052,0.497,0.465,0.830,0.588,0.506,0.419,0.565,0.095,0.309,0.180,0.601,0.912,0.617,0.722,0.824,0.651,0.434,0.418,0.657,0.661,0.602,0.143,0.679,0.527,0.927,0.029,0.117,0.428,0.857,0.632,0.707,0.585,0.037,0.703,0.281,0.516,0.942,0.062,0.068,0.416,0.366,0.358,0.955,0.940,0.725,0.945,0.267,0.488,0.633,0.633,0.310,0.333,0.623,0.235,0.072,0.243,0.027,0.586,0.463,0.865,0.331,0.150,0.133,0.276,0.710,0.483,0.116,0.563,0.810,0.892,0.592,0.753,0.848,0.813,0.280,0.606,0.068,0.392,0.889,0.921,0.270,0.007,0.005,0.775,0.348,0.281,0.395,0.719,0.279,0.782,0.619,0.578,0.407,0.423,0.473,0.371,0.277,0.174,0.473,0.414,0.690,0.885,0.440,0.413,0.574,0.167,0.687,0.420,0.261,0.661,0.039,0.626,0.992,0.089,0.714,0.436,0.983,0.302,0.847,0.466,0.671,0.786,0.995,0.581,0.377,0.231,0.864,0.666,0.366,0.612,0.671,0.607,0.688,0.520,0.335,0.872,0.533,0.063,0.739,0.720,0.727,0.887,0.094,0.512,0.058,0.251,0.467,0.674,0.855,0.108,0.994,0.151,0.338,0.028,0.922,0.483,0.067,0.155,0.631,0.170,0.623,0.794,0.542,0.615,0.970,0.435,0.415,0.464,0.945,0.634,0.837,0.992,0.372,0.625,0.218,0.305,0.782,0.884,0.878,0.218,0.218,0.573,0.518,0.348,0.622,0.646,0.095,0.357,0.163,0.948,0.047,0.729,0.955,0.744,0.428,0.945,0.971,0.341,0.190,0.575,0.615,0.092,0.520,0.097,0.632,0.880,0.279,0.302,0.381,0.199,0.676,0.601,0.379,0.827,0.599,0.019,0.166,0.340,0.147,0.418,0.350,0.773,0.953,0.576,0.631,0.950,0.827,0.450,0.763,0.518,0.176,0.933,0.046,0.449,0.100,0.584,0.749,0.876,0.788,0.054,0.590,0.235,0.321,0.012,0.357,0.625,0.147,0.274,0.611,0.136,0.391,0.079,0.044,0.425,0.419,0.156,0.819,0.140,0.835,0.650,0.682,0.969,0.819,0.317,0.950,0.152,0.017,0.692,0.187,0.350,0.252,0.953,0.118,0.666,0.698,0.792,0.744,0.805,0.087,0.447,0.893,0.386,0.296,0.453,0.108,0.829,0.027,0.593,0.880,0.777,0.476,0.244,0.245,0.439,0.156,0.973,0.513,0.982,0.183,0.737,0.462,0.822,0.500,0.172,0.301,0.467,0.206,0.341,0.601,0.200,0.808,0.693,0.236,0.490,0.797,0.298,0.767,0.947,0.252,0.031,0.980,0.621,0.864,0.955,0.357,0.294,0.238,0.619,0.991,0.499,0.648,0.852,0.380,0.316,0.786,0.316,0.276,0.203,0.711,0.481,0.995,0.383,0.128,0.220,0.897,0.572,0.731,0.384,0.697,0.123,0.084,0.519,0.034,0.723,0.849,0.640,0.304,0.788,0.746,0.064,0.565,0.639,0.152,0.492,0.825,0.180,0.047,0.024,0.193,0.249,0.539,0.413,0.103,0.156,0.413,0.898,0.756,0.936,0.914,0.200,0.185,0.505,0.277,0.701,0.745,0.343,0.634,0.054,0.637,0.538,0.032,0.071,0.471,0.302,0.321,0.368,0.532,0.557,0.328,0.096,0.389,0.104,0.779,0.651,0.084,0.202,0.205,0.927,0.912,0.292,0.843,0.989,0.983,0.729,0.137,0.130,0.615,0.068,0.946,0.541,0.177,0.409,0.876,0.087,0.082,0.955,0.717,0.574,0.056,0.764,0.774;

    workspace.A<<0.945,0.262,0.494,0.293,0.457,0.242,0.739,0.741,0.871,0.337,0.351,0.009,0.142,0.991,0.638,0.715,0.668,0.207,0.167,0.552,0.003,0.228,0.084,0.421,0.613,0.455,0.667,0.542,0.322,0.315,0.947,0.388,0.667,0.045,0.980,0.347,0.664,0.125,0.432,0.506,0.246,0.049,0.429,0.470,0.164,0.150,0.663,0.505,0.581,0.602,0.443,0.052,0.049,0.175,0.363,0.946,0.242,0.835,0.516,0.934,0.276,0.639,0.847,0.270,0.027,0.236,0.741,0.427,0.813,0.168,0.491,0.247,0.722,0.646,0.154,0.374,0.665,0.619,0.155,0.983,0.187,0.539,0.352,0.214,0.390,0.389,0.925,0.812,0.877,0.539,0.602,0.324,0.955,0.393,0.581,0.143,0.641,0.673,0.474,0.605,0.930,0.447,0.161,0.002,0.393,0.201,0.375,0.093,0.445,0.127,0.420,0.291,0.062,0.712,0.423,0.261,0.902,0.546,0.874,0.078,0.163,0.904,0.021,0.891,0.773,0.908,0.538,0.582,0.760,0.221,0.858,0.282,0.437,0.250,0.916,0.603,0.211,0.466,0.926,0.880,0.139,0.991,0.085,0.621,0.655,0.036,0.364,0.708,0.617,0.162,0.978,0.625,0.667,0.409,0.418,0.234,0.791,0.949,0.964,0.424,0.386,0.528,0.626,0.483,0.060,0.778,0.781,0.213,0.784,0.126,0.585,0.410,0.260,0.943,0.632,0.446,0.608,0.150,0.762,0.574,0.236,0.517,0.044,0.670,0.556,0.471,0.154,0.061,0.634,0.771,0.046,0.086,0.090,0.181,0.192,0.831,0.628,0.179,0.363,0.763,0.272,0.421,0.656,0.982,0.735,0.084,0.362,0.835,0.792,0.059,0.583,0.344,0.752,0.945,0.938,0.548,0.175,0.598,0.407,0.089,0.693,0.917,0.418,0.104,0.441,0.169,0.787,0.936,0.616,0.591,0.821,0.840,0.800,0.628,0.706,0.301,0.693,0.568,0.962,0.698,0.239,0.657,0.217,0.267,0.400,0.145,0.938,0.279,0.498,0.664,0.887,0.594,0.659,0.688,0.269,0.926,0.305,0.072,0.656,0.768,0.688,0.458,0.094,0.789,0.650,0.479,0.230,0.201,0.385,0.768,0.826,0.488,0.556,0.742,0.776,0.881,0.083,0.085,0.522,0.434,0.526,0.739,0.248,0.547,0.988,0.136,0.643,0.563,0.296,0.403,0.143,0.424,0.136,0.823,0.851,0.200,0.262,0.279,0.458,0.518,0.007,0.745,0.776,0.482,0.095,0.253,0.835,0.222,0.014,0.366,0.918,0.231,0.510,0.320,0.599,0.190,0.823,0.619,0.447,0.416,0.391,0.184,0.678,0.450,0.584,0.711,0.930,0.354,0.328,0.418,0.060,0.706,0.605,0.690,0.196,0.377,0.870,0.024,0.553,0.945,0.513,0.859,0.997,0.552,0.678,0.218,0.094,0.125,0.994,0.092,0.939,0.434,0.929,0.460,0.890,0.769,0.335,0.535,0.939,0.727,0.052,0.087,0.457,0.725,0.912,0.222,0.560,0.906,0.768,0.308,0.890,0.538,0.036,0.333,0.962,0.211,0.457,0.118,0.004,0.102,0.514,0.683,0.662,0.092,0.433,0.534,0.424,0.332,0.899,0.783,0.297,0.815,0.358,0.413,0.561,0.022,0.093,0.933,0.751,0.352,0.731,0.168,0.967,0.380,0.456,0.417,0.249,0.094,0.918,0.443,0.462,0.476,0.127,0.696,0.897,0.867,0.030,0.376,0.730,0.452,0.968,0.530,0.982,0.584,0.518,0.166,0.149,0.769,0.301,0.290,0.469,0.973,0.496,0.641,0.057,0.373,0.835,0.982,0.952,0.758,0.070,0.432,0.335,0.310,0.814,0.627,0.894,0.178,0.339,0.146,0.851,0.237,0.968,0.945,0.461,0.795,0.842,0.329,0.347,0.360,0.902,0.266,0.613,0.456,0.830,0.555,0.525,0.907,0.116,0.046,0.841,0.192,0.287,0.833,0.591,0.618,0.173,0.841,0.756,0.919,0.612,0.037,0.854,0.656,0.712,0.161,0.095,0.279,0.506,0.283,0.368,0.123,0.939,0.269,0.429,0.657,0.825,0.606,0.094,0.218,0.084,0.759,0.937,0.654,0.754,0.712,0.306,0.307,0.838,0.319,0.837,0.795,0.378,0.476,0.555,0.843,0.416,0.074,0.929,0.198,0.086,0.400,0.607,0.873,0.584,0.710,0.199,0.687,0.598,0.450,0.435,0.117,0.410,0.604,0.039,0.931,0.177,0.460,0.313,0.440,0.217,0.462,0.107,0.259,0.113,0.276,0.579,0.692,0.184,0.061,0.509,0.374,0.349,0.891,0.301,0.426,0.518,0.431,0.841,0.228,0.128,0.739,0.993,0.197,0.623,0.377,0.831,0.238,0.654,0.374,0.160,0.672,0.546,0.772,0.981,0.254,0.036,0.023,0.121,0.008,0.413,0.449,0.293,0.440,0.115,0.410,0.285,0.549,0.300,0.558,0.915,0.208,0.434,0.042,0.093,0.171,0.530,0.550,0.678,0.298,0.471,0.044,0.457,0.406,0.415,0.106,0.192,0.620,0.208,0.072,0.894,0.940,0.736,0.730,0.936,0.675,0.012,0.311,0.528,0.796,0.614,0.906,0.012,0.885,0.108,0.685,0.581,0.573,0.205,0.313,0.924,0.376,0.823,0.274,0.943,0.592,0.758,0.839,0.341,0.736,0.617,0.172,0.963,0.492,0.133,0.598,0.977,0.304,0.565,0.699,0.425,0.319,0.150,0.982,0.989,0.449,0.501,0.430,0.011,0.216,0.591,0.379,0.059,0.231,0.607,0.670,0.250,0.844,0.077,0.373,0.956,0.323,0.923,0.286,0.889,0.021,0.353,0.672,0.836,0.510,0.860,0.778,0.652,0.567,0.939,0.698,0.969,0.281,0.801,0.516,0.898,0.937,0.244,0.782,0.128,0.062,0.016,0.820,0.600,0.876,0.240,0.197,0.129,0.405,0.543,1.000,0.556,0.881,0.127,0.771,0.230,0.147,0.242,0.608,0.446,0.223,0.101,0.018,0.436,0.904,0.902,0.371,0.447,0.402,0.206,0.133,0.300,0.122,0.789,0.142,0.558,0.644,0.104,0.706,0.884,0.634,0.960,0.206,0.457,0.898,0.569,0.435,0.504,0.524,0.621,0.526,0.762,0.837,0.686,0.969,0.091,0.066,0.395,0.919,0.574,0.476,0.019,0.002,0.616,0.956,0.382,0.082,0.420,0.888,0.660,0.933,0.883,0.402,0.870,0.491,0.177,0.951,0.430,0.328,0.313,0.756,0.934,0.383,0.485,0.143,0.892,0.968,0.947,0.581,0.492,0.314,0.862,0.415,0.945,0.653,0.392,0.672,0.321,0.714,0.010,0.570,0.649,0.027,0.242,0.805,0.725,0.091,0.105,0.011,0.382,0.412,0.762,0.390,0.666,0.470,0.872,0.796,0.171,0.720,0.166,0.142,0.350,0.138,0.695,0.699,0.786,0.541,0.190,0.339,0.818,0.277,0.001,0.919,0.821,0.234,0.589,0.418,0.802,0.925,0.065,0.427,0.450,0.157,0.907,0.082,0.035,0.470,0.006,0.605,0.490,0.021,0.806,0.637,0.812,0.890,0.181,0.876,0.011,0.658,0.854,0.683,0.660,0.736,0.077,0.027,0.327,0.884,0.908,0.881,0.916,0.850,0.437,0.037,0.626,0.628,0.958,0.715,0.482,0.411,0.030,0.410,0.945,0.615,0.830,0.363,0.238,0.512,0.240,0.629,0.308,0.739,0.960,0.099,0.504,0.284,0.019,0.633,0.029,0.183,0.203,0.504,0.038,0.254,0.555,0.304,0.439,0.755,0.848,0.161,0.369,0.276,0.036,0.089,0.122,0.538,0.664,0.046,0.999,0.482,0.460,0.268,0.121,0.654,0.068,0.863,0.049,0.953,0.632,0.788,0.771,0.308,0.124,0.753,0.104,0.399,0.768,0.033,0.616,0.047,0.635,0.978,0.411,0.084,0.411,0.789,0.782,0.484,0.266,0.596,0.776,0.817,0.941,0.305,0.250,0.345,0.505,0.882,0.316,0.272,0.200,0.422,0.435,0.571,0.380,0.547,0.658,0.442,0.580,0.433,0.099,0.012,0.418,0.660,0.409,0.078,0.988,0.712,0.657,0.943,0.992,0.066,0.202,0.076,0.237,0.713,0.021,0.387,0.278,0.075,0.781,0.309,0.988,0.787,0.154,0.855,0.170,0.875,0.274,0.706,0.579,0.860,0.266,0.931,0.842,0.412,0.366,0.056,0.025,0.897,0.898,0.399,0.907,0.088,0.725,0.950,0.173,0.933,0.053,0.096,0.688,0.306,0.893,0.176,0.944,0.999,0.099,0.520,0.242,0.179,0.824,0.681,0.029,0.064,0.771,0.712,0.167,0.087,0.668,0.245,0.180,0.491,0.062,0.550,0.161,0.767,0.448,0.067,0.092,0.784,0.802,0.293,0.308,0.523,0.345,0.293,0.214,0.209,0.801,0.509,0.160,0.292,0.460,0.813,0.817,0.389,0.492,0.385,0.414,0.394,0.351,0.363,0.084,0.148,0.176,0.303,0.283,0.591,0.611,0.746,0.273,0.570,0.657,0.010,0.168,0.631,0.698,0.054,0.204,0.995,0.344,0.833,0.729,0.509,0.516,0.954,0.784,0.298,0.031,0.297,0.037,0.503,0.571,0.429,0.426,0.681,0.276,0.658,0.974,0.675,0.876,0.395,0.581,0.955,0.813,0.381,0.477,0.697,0.242,0.806,0.663,0.082,0.805,0.593,0.506,0.002,0.729,0.827,0.953,0.208,0.138,0.715,0.104,0.580,0.972,0.635,0.887,0.129,0.662,0.286,0.580,0.393,0.581,0.307,0.717,0.186,0.368,0.111,0.605,0.201,0.836,0.383,0.246,0.886,0.548,0.316,0.742,0.544,0.330,0.437,0.075,0.562,0.435,0.480,0.898,0.873,0.889,0.597,0.795,0.876,0.174,0.497,0.261,0.171,0.262,0.575,0.691,0.310,0.717,0.606,0.332,0.473,0.777,0.519,0.660,0.541,0.245,0.189,0.820,0.908,0.799,0.561,0.896,0.521,0.623,0.551,0.508,0.680,0.508,0.533,0.989,0.454,0.157,0.198,0.244,0.841,0.098,0.608,0.547,0.495,0.732,0.002,0.072,0.844,0.139,0.279,0.501,0.284,0.783,0.145,0.503,0.925,0.788,0.191,0.894,0.516,0.791,0.398,0.255,0.251,0.816,0.934,0.541,0.815,0.285,0.634,0.763,0.818,0.035,0.039,0.885,0.965,0.081,0.407,0.277,0.814,0.698,0.571,0.650,0.556,0.088,0.661,0.736,0.911,0.264,0.893,0.779,0.448,0.826,0.397,0.543,0.959,0.104,0.298,0.675,0.931,0.671,0.038,0.936,0.982,0.234,0.738,0.498,0.672,0.346,0.897,0.855,0.578,0.722,0.120,0.541,0.537,0.123,0.059,0.754,0.692,0.367,0.289,0.749,0.918,0.494,0.860,0.629,0.686,0.797,0.321,0.850,0.204,0.325,0.933,0.878,0.934,0.617,0.627,0.495,0.676,0.731,0.870,0.753,0.719,0.636,0.644,0.924,0.589,0.651,0.223,0.895,0.761,0.535,0.183,0.798,0.201,0.651,0.546,0.141,0.314,0.022,0.172,0.146,0.002,0.983,0.723,0.077,0.352,0.043,0.791,0.901,0.386,0.315,0.447,0.114,0.229,0.051,0.465,0.734,0.995,0.131,0.792,0.182,0.054,0.216,0.805,0.050,0.489,0.844,0.340,0.466,0.095,0.010,0.733,0.476,0.982,0.863,0.656,0.125,0.860,0.431,0.731,0.169,0.179,0.158,0.412,0.828,0.260,0.143,0.031,0.029,0.538,0.739,0.444,0.349,0.339,0.548,0.583,0.718,0.471,0.344,0.327,0.629,0.912,0.668,0.335,0.028,0.906,0.999,0.938,0.356,0.618,0.520,0.218,0.992,0.692,0.119,0.812,0.606,0.998,0.946,0.186,0.421,0.612,0.835,0.192,0.252,0.952,0.317,0.349,0.385,0.086,0.440,0.739,0.374,0.098,0.495,0.032,0.750,0.829,0.580,0.466,0.558,0.221,0.767,0.722,0.720,0.622,0.847,0.005,0.080,0.367,0.075,0.862,0.254,0.833,0.128,0.335,0.064,0.492,0.969,0.503,0.064,0.325,0.821,0.234,0.068,0.903,0.605,0.139,0.682,0.029,0.733,0.100,0.837,0.125,0.372,0.331,0.501,0.341,0.050,0.559,0.165,0.933,0.656,0.007,0.075,0.786,0.128,0.972,0.563,0.285,0.457,0.034,0.536,0.859,0.513,0.057,0.780,0.561,0.631,0.751,0.488,0.339,0.505,0.040,0.396,0.509,0.763,0.093,0.952,0.750,0.244,0.094,0.783,0.044,0.331,0.084,0.719,0.241,0.107,0.204,0.661,0.687,0.936,0.894,0.157,0.389,0.925,0.735,0.898,0.288,0.183,0.503,0.905,0.977,0.331,0.402,0.543,0.779,0.886,0.011,0.433,0.633,0.569,0.618,0.041,0.376,0.398,0.388,0.590,0.297,0.803,0.341,0.486,0.250,0.492,0.967,0.448,0.314,0.295,0.679,0.875,0.399,0.115,0.920,0.318,0.654,0.357,0.132,0.686,0.262,0.853,0.069,0.614,0.530,0.301,0.876,0.413,0.343,0.347,0.089,0.314,0.196,0.444,0.009,0.756,0.578,0.725,0.394,0.937,0.003,0.213,0.547,0.508,0.557,0.503,0.379,0.813,0.630,0.488,0.608,0.825,0.417,0.930,0.263,0.582,0.932,0.887,0.906,0.780,0.978,0.943,0.299,0.337,0.236,0.464,0.272,0.256,0.671,0.968,0.218,0.594,0.406,0.219,0.251,0.751,0.919,0.734,0.192,0.726,0.969,0.898,0.279,0.617,0.572,0.194,0.431,0.873,0.004,0.793,0.694,0.732,0.298,0.054,0.058,0.718,0.394,0.025,0.248,0.625,0.044,0.712,0.010,0.526,0.706,0.415,0.823,0.137,0.878,0.667,0.817,0.869,0.165,0.607,0.850,0.193,0.395,0.937,0.946,0.045,0.708,0.171,0.076,0.024,0.543,0.998,0.233,0.904,0.338,0.544,0.368,0.091,0.730,0.462,0.185,0.799,0.733,0.311,0.226,0.182,0.252,0.804,0.058,0.037,0.983,0.230,0.886,0.907,0.242,0.955,0.355,0.257,0.469,0.660,0.529,0.557,0.312,0.637,0.451,0.682,0.840,0.457,0.342,0.470,0.206,0.097,0.898,0.745,0.841,0.419,0.871,0.977,0.319,0.593,0.484,0.693,0.915,0.282,0.127,0.873,0.564,0.267,0.420,0.157,0.324,0.598,0.740,0.286,0.329,0.793,0.823,0.912,0.716,0.721,0.007,0.435,0.112,0.044,0.990,0.322,0.804,0.669,0.750,0.373,0.925,0.226,0.950,0.082,0.576,0.768,0.186,0.902,0.701,0.318,0.506,0.004,0.131,0.706,0.689,0.215,0.429,0.175,0.518,0.651,0.101,0.221,0.495,0.901,0.758,0.058,0.252,0.073,0.300,0.178,0.217,0.941,0.189,0.105,0.039,0.823,0.566,0.918,0.621,0.675,0.936,0.769,0.642,0.632,0.408,0.527,0.734,0.858,0.322,0.895,0.396,0.619,0.938,0.246,0.861,0.809,0.558,0.850,0.088,0.286,0.375,0.478,0.551,0.114,0.445,0.302,0.581,0.858,0.238,0.874,0.009,0.272,0.316,0.065,0.253,0.265,0.998,0.688,0.704,0.231,0.966,0.152,0.467,0.750,0.856,0.899,0.435,0.645,0.051,0.393,0.828,0.128,0.584,0.339,0.446,0.730,0.210,0.424,0.304,0.700,0.634,0.540,0.714,0.478,0.868,0.198,0.725,0.232,0.700,0.059,0.706,0.897,0.678,0.045,0.014,0.930,0.047,0.015,0.445,0.006,0.281,0.039,0.122,0.190,0.816,0.366,0.420,0.263,0.398,0.213,0.009,0.941,0.404,0.070,0.457,0.861,0.732,0.581,0.280,0.149,0.064,0.465,0.804,0.060,0.105,0.996,0.977,0.175,0.899,0.955,0.341,0.692,0.798,0.262,0.913,0.692,0.693,0.945,0.101,0.848,0.108,0.645,0.095,0.042,0.746,0.652,0.784,0.588,0.797,0.321,0.929,0.545,0.054,0.349,0.225,0.287,0.119,0.256,0.966,0.946,0.882,0.245,0.442,0.743,0.945,0.941,0.986,0.294,0.331,0.160,0.856,0.144,0.634,0.086,0.766,0.789,0.831,0.673,0.343,0.729,0.758,0.609,0.732,0.941,0.322,0.603,0.871,0.811,0.615,0.998,0.289,0.233,0.849,0.034,0.655,0.939,0.000,0.810,0.086,0.207,0.337,0.393,0.936,0.839,0.097,0.821,0.836,0.265,0.192,0.028,0.417,0.164,0.459,0.391,0.063,0.136,0.361,0.726,0.619,0.200,0.021,0.090,0.325,0.208,0.580,0.696,0.643,0.626,0.862,0.745,0.997,0.850,0.189,0.987,0.888,0.043,0.733,0.951,0.750,0.881,0.305,0.897,0.057,0.610,0.602,0.149,0.829,0.065,0.866,0.712,0.193,0.253,0.970,0.533,0.088,0.308,0.240,0.407,0.061,0.099,0.877,0.509,0.171,0.535,0.512,0.068,0.973,0.749,0.993,0.074,0.180,0.473,0.626,0.654,0.545,0.657,0.380,0.446,0.731,0.498,0.338,0.725,0.821,0.720,0.997,0.073,0.281,0.032,0.018,0.295,0.021,0.619,0.922,0.266,0.653,0.824,0.387,0.638,0.448,0.271,0.002,0.097,0.258,0.180,0.821,0.781,0.745,0.088,0.807,0.320,0.857,0.206,0.556,0.226,0.562,0.678,0.824,0.007,0.682,0.633,0.569,0.208,0.979,0.022,0.896,0.214,0.470,0.071,0.155,0.752,0.891,0.048,0.436,0.729,0.857,0.550,0.534,0.775,0.405,0.066,0.125,0.664,0.254,0.611,0.302;

    workspace.q<<0.827,0.334,0.359,0.560,0.124,0.153,0.373,0.997,0.631,0.233,0.959,0.797,0.203,0.463,0.754,0.774,0.463,0.376,0.561,0.860,0.084,0.592,0.700,0.103,0.896,0.211,0.604,0.411,0.678,1.000;

    workspace.lower<<0.799,0.890,0.460,0.307,0.196,0.223,0.765,0.532,0.173,0.562,0.558,0.836,0.644,0.362,0.970,0.702,0.704,0.310,0.359,0.378,0.416,0.485,0.160,0.189,0.448,0.534,0.219,0.772,0.762,0.514,0.523,0.677,0.652,0.560,0.330,0.729,0.942,0.985,0.098,0.538,0.828,0.005,0.324,0.240,0.074,0.046,0.335,0.527,0.065,0.276,0.361,0.220,0.890,0.226,0.988,0.385,0.671,0.055,0.948,0.936,0.651,0.517,0.307,0.234,0.786,0.898,0.438,0.480,0.207,0.499;
    workspace.upper<<1.398,1.020,1.153,1.958,1.012,1.261,1.302,1.582,1.996,1.233,1.162,1.874,1.363,1.609,1.734,1.539,1.004,1.598,1.724,1.888,1.360,1.790,1.056,1.026,1.860,1.461,1.736,1.592,1.420,1.147,1.057,1.973,1.315,1.632,1.635,1.545,1.207,1.974,1.169,1.438,1.318,1.304,1.154,1.830,1.597,1.810,1.152,1.323,1.563,1.484,1.015,1.282,1.247,1.633,1.637,1.186,1.146,1.844,1.027,1.936,1.710,1.516,1.518,1.360,1.118,1.041,1.762,1.081,1.133,1.103,

    workspace.At << workspace.A.transpose();
    workspace.KKT<<workspace.P,workspace.At, workspace.A,workspace.rho_inv_mat;
    workspace.KKT_tmp = workspace.KKT;

//    update_matrix(workspace, P, A, q, upper, lower);
    TICK(timer)
    bool status = admm(workspace);
    if(status){
        polish(workspace);
        std::cout<<"soltions:\n"<<workspace.x<<std::endl;
    }
    TOCK(timer)
}