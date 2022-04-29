#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include "macros.h"
#include <cstdio>
#include <iostream>
#include "helper_cuda.h"
#include <string>
#include "functions.h"
/*-------------------------------------------------------------------------------------*/
typedef struct{
    float* P;
    float* A;
    float* At;
    float* KKT;
    float* KKT_tmp;
    float* rho_vec;
    float* rho_inv_mat;
    float* negative_rho_inv_mat;
    float* rho_inv_vec;
    float* rhs;
    float* rhs_prev;
    float* x, *y, *z, *solve_tmp;
    float* x_prev, *y_prev, *z_prev;
    float* x_delta, *y_delta, *z_delta;
    float* Ax;
    float* q, *upper, *lower;
    /*----------scalars----------*/
    float* transpose_a, *transpose_b;
    float* d_alpha, alpha;
    float* d_sigma, sigma;
    float* d_rho, rho;
    float* d_rho_estimate, rho_estimate;
    float* d_adapt_rho_tol, adapt_rho_tol;
    float* d_eps_pri_limit, eps_pri_limit;
    float* d_eps_dual_limit, eps_dual_limit;
    float* d_eps_pinf_limit, eps_pinf_limit;
    float* d_eps_dinf_limit, eps_dinf_limit;
    float *d_pri_res,pri_res;
    float *d_dua_res, dua_res;
    float *P_buffer;
    int m;
    int n;
    bool compute_objective;
    float obj_val;

    /*-------cusolver-----*/
    int bufferSize;
    int *info;
    float *buffer;
    float *tau;
    /*----------handles----------*/
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverDnHandle;
} workspace;
/*-------------------------------------------------------------------------------------*/
void workspace_init(workspace *workspace){
    int m = workspace->m, n = workspace->n;
    int size = m + n;
    checkCudaErrors(cudaMalloc((void**)&workspace->P, sizeof(float) * n * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->A, sizeof(float) * m * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->At, sizeof(float) * m * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->KKT, sizeof(float) * size * size));
    checkCudaErrors(cudaMalloc((void**)&workspace->KKT_tmp, sizeof(float) * size * size));

    checkCudaErrors(cudaMalloc((void**)&workspace->q, sizeof(float) * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->upper, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->lower, sizeof(float) * m));

    checkCudaErrors(cudaMalloc((void**)&workspace->rho_vec, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->rho_inv_mat, sizeof(float) * m * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->negative_rho_inv_mat, sizeof(float) * m * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->rho_inv_vec, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->rhs, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void**)&workspace->rhs_prev, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void**)&workspace->x, sizeof(float) * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->x_prev, sizeof(float) * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->y, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->y_prev, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->z, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->z_prev, sizeof(float) * m));

    checkCudaErrors(cudaMalloc((void**)&workspace->x_delta, sizeof(float) * n));
    checkCudaErrors(cudaMalloc((void**)&workspace->y_delta, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->z_delta, sizeof(float) * m));
    checkCudaErrors(cudaMalloc((void**)&workspace->solve_tmp, sizeof(float) * size));
    checkCudaErrors(cudaMalloc((void**)&workspace->Ax, sizeof(float) * m));

    checkCudaErrors(cudaMalloc((void**)&workspace->d_alpha, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_sigma, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_rho, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_adapt_rho_tol, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_rho_estimate, sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&workspace->d_pri_res, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_dua_res, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_eps_pri_limit, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_eps_dual_limit, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_eps_pinf_limit, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->d_eps_dinf_limit, sizeof(float)));
    
    workspace->bufferSize = -1;
    cudaMalloc((void**)&workspace->info, (m + n) * sizeof(float));
    cudaMalloc((void**)&workspace->tau, sizeof(float)*(m + n));
    /*----------init handles--------------*/
    checkCudaErrors(cudaMalloc((void**)&workspace->transpose_a, sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&workspace->transpose_b, sizeof(float)));
    checkCudaErrors(cudaMemset(workspace->transpose_a, 1, sizeof(float )));
    checkCudaErrors(cudaMemset(workspace->transpose_b, 1, sizeof(float )));


    checkCudaErrors(cudaMalloc((void**)&workspace->P_buffer, n*sizeof(float)));
    checkCudaErrors(cudaMemset(workspace->P_buffer, 0, n * sizeof(float )));
    checkCudaErrors(cublasCreate_v2(&workspace->cublasHandle)); // 350ms
    checkCudaErrors(cusolverDnCreate(&workspace->cusolverDnHandle)); //400ms
    printf("INITIALIZATION FINISHED. \n");
}

void workspace_default_setting(workspace *workspace){
    int m = workspace->m, n = workspace->n;
    workspace->eps_dual_limit = (float)MAX_TOL;
    workspace->eps_pri_limit = (float)MAX_TOL;
    workspace->eps_pinf_limit = (float)MAX_TOL;
    workspace->eps_dinf_limit = (float)MAX_TOL;
    workspace->compute_objective = COMPUTE_OBJ;
    workspace->obj_val = -1.0;
    workspace->pri_res = 0.0;
    workspace->dua_res = 0.0;
    workspace->alpha = ALPHA;
    workspace->sigma = 1e-6;
    workspace->rho = RHO;
    checkCudaErrors(cudaMemcpy(workspace->d_eps_dual_limit, &workspace->eps_dual_limit, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_eps_pri_limit, &workspace->eps_pri_limit, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_eps_pinf_limit, &workspace->eps_pinf_limit, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_eps_dinf_limit, &workspace->eps_dinf_limit, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_pri_res, &workspace->pri_res, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_dua_res, &workspace->dua_res, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_alpha, &workspace->alpha, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_sigma, &workspace->sigma, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->d_rho, &workspace->rho, sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(workspace->rho_vec, workspace->rho, m * sizeof(float)));
    checkCudaErrors(cudaMemset(workspace->rho_inv_vec, 1/workspace->rho, m * sizeof(float)));


    std::vector<float> r( m * m, 0.0);
    for(int i = 0; i < m; i++){
        r[i * m + i] = 1.0 / workspace->rho;
    }
    checkCudaErrors(cudaMemcpy(workspace->rho_inv_mat, r.data(), sizeof(float) * m * m, cudaMemcpyHostToDevice));

    for(int i = 0; i < m; i++){
        r[i * m + i] = -1.0 / workspace->rho;
    }
    checkCudaErrors(cudaMemcpy(workspace->negative_rho_inv_mat, r.data(), sizeof(float) * m * m, cudaMemcpyHostToDevice));
    printf("DEFAULT SETTING FINISHED. \n");
}
void cuda_vec_add_scaled(workspace *workspace,
                        float       *d_x,
                         const float *d_a,
                         const float *d_b,
                         float        sca,
                         float        scb,
                         int          n) {

    if (d_x != d_a || sca != 1.0) {
        if (sca == 1.0) {
            /* d_x = d_a */
            checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        else if (d_x == d_a) {
            /* d_x *= sca */
            checkCudaErrors(cublasSscal_v2(workspace->cublasHandle, n, &sca, d_x, 1));
        }
        else {
            /* d_x = 0 */
            checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(float)));

            /* d_x += sca * d_a */
            checkCudaErrors(cublasSaxpy_v2(workspace->cublasHandle, n, &sca, d_a, 1, d_x, 1));
        }
    }

    /* d_x += scb * d_b */
    checkCudaErrors(cublasSaxpy_v2(workspace->cublasHandle, n, &scb, d_b, 1, d_x, 1));
}

__global__ void cuda_merge_matrix(float* dst, float* src, int dst_m, int dst_n,
                                  int src_m, int src_n, int offset_x, int offset_y){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = id / src_n, col = id % src_n;
    int pos = (row+offset_x) * dst_n + (col+offset_y);
    dst[pos] = src[id];
}

__global__ void vec_ew_prod_kernel(float       *c,
                                   const float *a,
                                   const float *b,
                                   int          n) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += grid_size) {
        c[i] = __fmul_rn(a[i], b[i]);
    }
}
void cuda_vec_ew_prod(float       *d_c,
                      const float *d_a,
                      const float *d_b,
                      int          n) {

    int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

    vec_ew_prod_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, n);
}
__global__ void vec_bound_kernel(float       *x,
                                 const float *z,
                                 const float *l,
                                 const float *u,
                                 int          n) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_size = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += grid_size) {
        x[i] = min(max(z[i], l[i]), u[i]);
    }
}
void cuda_vec_bound(float       *d_x,
                    const float *d_z,
                    const float *d_l,
                    const float *d_u,
                    int          n) {

    int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

    vec_bound_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(d_x, d_z, d_l, d_u, n);
}

float cuda_vec_norm_inf(workspace * workspace, float *a, int n){
    int idx;
    float res;
    checkCudaErrors(cublasIsamax_v2(workspace->cublasHandle, n, a,1, &idx));
    checkCudaErrors(cudaMemcpy(&res, a + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return res;
}

void print_matrix(std::string name, float* ptr, int m, int n=1){
    size_t size = sizeof(float) * m * n;
    float *mat = (float*)malloc(sizeof(float) * m * n);
    checkCudaErrors(cudaMemcpy(mat, ptr, size, cudaMemcpyDeviceToHost));
    for(int i = 0; i < m; i++){
        for(int j =0; j<n;j++){
            printf("%s[%d][%d]: %f ",name.c_str(), i, j, mat[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}
void cuda_mat_mul(workspace* workspace, float* A, float *B, float *C, int m, int n, int k, int lda, int ldb, int ldc, float alpha=1.0, float beta = 0.0){
//    float alpha = 1.0, beta = 0.0;
    checkCudaErrors(cublasSgemm(workspace->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

//void cuda_mat_vec_mul(workspace *workspace, const float* A, const float* B, float *C, int m ,int n, int kl, int ku, int lda
//                      , int incb = 1, int incc = 1, float alpha = 1.0, float beta = 0.0){
////    float alpha = 1.0, beta = 0.0;
//    float alpha_t = alpha;
//    float beta_t = beta;
//    checkCudaErrors(cublasSgbmv_v2(workspace->cublasHandle, CUBLAS_OP_T, m, n, kl, ku, &alpha_t, A, lda, B, incb, &beta_t, C, incc));
//}
void cuda_mat_transpose(workspace* workspace, float* A, float* At, int lda, int ldb){
    int m = workspace->m, n = workspace->n;
    float alpha = 1.0, beta = 0.0;
    cublasSgeam(workspace->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                m, n, &alpha, A, ldb, &beta, A, ldb, At, lda);
}
void cuda_vec_add_scaled3(workspace *workspace,
                            float       *d_x,
                          const float *d_a,
                          const float *d_b,
                          const float *d_c,
                          float        sca,
                          float        scb,
                          float        scc,
                          int          n) {

    if (d_x != d_a || sca != 1.0) {
        if (sca == 1.0) {
            /* d_x = d_a */
            checkCudaErrors(cudaMemcpy(d_x, d_a, n * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        else if (d_x == d_a) {
            /* d_x *= sca */
            checkCudaErrors(cublasSscal_v2(workspace->cublasHandle, n, &sca, d_x, 1));
        }
        else {
            /* d_x = 0 */
            checkCudaErrors(cudaMemset(d_x, 0, n * sizeof(float)));

            /* d_x += sca * d_a */
            checkCudaErrors(cublasSaxpy_v2(workspace->cublasHandle, n, &sca, d_a, 1, d_x, 1));
        }
    }

    /* d_x += scb * d_b */
    checkCudaErrors(cublasSaxpy_v2(workspace->cublasHandle, n, &scb, d_b, 1, d_x, 1));

    /* d_x += scc * d_c */
    checkCudaErrors(cublasSaxpy_v2(workspace->cublasHandle, n, &scc, d_c, 1, d_x, 1));
}
float cuda_vec_prod(workspace *workspace, float *a, float* b, int n){
    float res;
    checkCudaErrors(cublasSdot_v2(workspace->cublasHandle, n, a, 1, b, 1, &res));
    return res;
}

float cuda_quad_form(workspace *workspace, float* P, float* x, int m ,int n){

    float res;
    cuda_mat_mul(workspace, workspace->P, workspace->x, workspace->P_buffer, m, 1, n, n, 1, 1);
    res += cuda_vec_prod(workspace, workspace->x, workspace->P_buffer, n);
    res *= 0.5;
    return res;
}



void matrix_update(workspace *workspace, std::vector<float> &P, std::vector<float> &A,
                   std::vector<float> &q, std::vector<float> &upper, std::vector<float> &lower){
    int m  = workspace->m, n = workspace->n;
    int size = m + n;
    checkCudaErrors(cudaMemcpy(workspace->P, P.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->A, A.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->q, q.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->upper, upper.data(), sizeof(float) * m, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(workspace->lower, lower.data(), sizeof(float) * m, cudaMemcpyHostToDevice));
    cuda_mat_transpose(workspace, workspace->A, workspace->At, m, n);
    cuda_merge_matrix<<<n, n>>>(workspace->KKT, workspace->P, size, size, n, n, 0, 0);
    cuda_merge_matrix<<<m, n>>>(workspace->KKT, workspace->A, size, size, m, n, n, 0);
    cuda_merge_matrix<<<m, n>>>(workspace->KKT, workspace->At, size, size, n, m, 0, n);
    cuda_merge_matrix<<<m, m>>>(workspace->KKT, workspace->negative_rho_inv_mat, size, size, m, m, n, n);
    checkCudaErrors(cudaMemcpy(workspace->KKT_tmp, workspace->KKT, sizeof(float)* size * size, cudaMemcpyDeviceToDevice));
    printf("UPDATE FINISHED. \n");
}
int linearSolverQR(workspace *workspace){

    const float one = 1.0;
    int n = workspace->m + workspace->n;
    float *Acopy = workspace->KKT_tmp;
    int lda = n;
    const float *b = workspace->rhs;
    float *x = workspace->solve_tmp;
    if(workspace->bufferSize<0){
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(workspace->cusolverDnHandle, n, n, Acopy, lda, &workspace->bufferSize));
        checkCudaErrors(cudaMalloc(&workspace->buffer, sizeof(float)*(workspace->bufferSize)));
    }

    checkCudaErrors(cudaMemset(workspace->info, 0, sizeof(int)));


// compute QR factorization
    CUSOLVER_CHECK(cusolverDnSgeqrf(workspace->cusolverDnHandle, n, n, (float*)Acopy, lda, (float*)workspace->tau,
                                    (float*)workspace->buffer, workspace->bufferSize, workspace->info));

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    CUSOLVER_CHECK(cusolverDnSormqr(
            workspace->cusolverDnHandle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_OP_T,
            n,
            1,
            n,
            Acopy,
            lda,
            workspace->tau,
            x,
            n,
            workspace->buffer,
            workspace->bufferSize,
            workspace->info));

    // x = R \ Q^T*b
    CUBLAS_CHECK(cublasStrsm(
            workspace->cublasHandle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n,
            1,
            &one,
            (float*)Acopy,
            lda,
            (float*)x,
            n));
//    end = clock();
//    printf("Solve: %f \n", (float)(end - start) / CLOCKS_PER_SEC);
    return 0;
}


void clean_up(workspace *workspace){
    checkCudaErrors(cudaFree(workspace->P));
    checkCudaErrors(cudaFree(workspace->A));
    checkCudaErrors(cudaFree(workspace->At));
    checkCudaErrors(cudaFree(workspace->KKT));
    checkCudaErrors(cudaFree(workspace->KKT_tmp));
    checkCudaErrors(cudaFree(workspace->q));
    checkCudaErrors(cudaFree(workspace->rho_inv_vec));
    checkCudaErrors(cudaFree(workspace->rho_vec));
    checkCudaErrors(cudaFree(workspace->rho_inv_mat));
    checkCudaErrors(cudaFree(workspace->d_rho_estimate));
    checkCudaErrors(cudaFree(workspace->rho_inv_vec));
    checkCudaErrors(cudaFree(workspace->rho_inv_vec));
    checkCudaErrors(cudaFree(workspace->rho_inv_vec));

    cublasDestroy(workspace->cublasHandle);
    cusolverDnDestroy(workspace->cusolverDnHandle);
    printf("CLEANUP FINISHED. \n");
}


void compute_rhs(workspace *workspace){
    int n = workspace->n, m = workspace->m;
    /*----------related to x_tilde----------*/
    cuda_vec_add_scaled(workspace, workspace->rhs, workspace->rhs_prev, workspace->q, workspace->sigma, -1, n);
    /*---------related to z_tilde-----------*/
    cuda_vec_add_scaled(workspace, workspace->rhs+n, workspace->rhs_prev+n, workspace->y, 1, -1/workspace->rho, m);
}

void update_xztilde(workspace *workspace){
    //results stored in rhs//
    compute_rhs(workspace);
    linearSolverQR(workspace);
    int n = workspace->n, m = workspace->m;
    checkCudaErrors(cudaMemcpy(workspace->rhs, workspace->solve_tmp, sizeof(float)*(m+n), cudaMemcpyDeviceToDevice));
    cuda_mat_mul(workspace,workspace->A, workspace->rhs, workspace->rhs+n, m, 1, n, n, 1, m);
}

void update_x(workspace* workspace){
    float *x = workspace->x;
    /*-----------update x----------*/
    cuda_vec_add_scaled(workspace, x, workspace->rhs, workspace->rhs_prev, workspace->alpha, (1-workspace->alpha), workspace->n);
    /*-----------update x----------*/
    cuda_vec_add_scaled(workspace, workspace->x_delta, x, workspace->rhs_prev, 1.0, -1.0, workspace->n);
}

void update_z(workspace* workspace){
    float *z = workspace->z;
    int n = workspace->n;
    cuda_vec_add_scaled(workspace, z, workspace->rhs+n, workspace->rhs_prev+n, workspace->alpha, (1.0-workspace->alpha), workspace->m);
    cuda_vec_add_scaled(workspace, z, z, workspace->y, 1.0, 1.0/workspace->rho, workspace->m);

    /*------project------*/
    cuda_vec_bound(z, z, workspace->lower, workspace->upper, workspace->m);
}
void update_y(workspace* workspace){
    float *y = workspace->y;
    int n = workspace->n;
    int m = workspace->m;
    /*------------update y_delta----------*/
    cuda_vec_add_scaled3(workspace, workspace->y_delta, workspace->rhs+n, workspace->rhs_prev+n, workspace->z,
                         workspace->alpha, 1.0-workspace->alpha, -1.0, m);
    cuda_vec_add_scaled(workspace, workspace->y_delta, workspace->y_delta, workspace->y_delta, workspace->rho, 0, m);
    /*-------------update y-------------*/
    cuda_vec_add_scaled(workspace, y, y, workspace->y_delta, 1.0, 1.0, m);
}
float compute_obj_val(workspace *workspace){
    float res;
    int m = workspace->m, n = workspace->n;
    res = cuda_quad_form(workspace, workspace->P, workspace->x, n, n) +
            cuda_vec_prod(workspace, workspace->q, workspace->x, n);

    return res;
}
void compute_kl_ku(int m, int n ,int &kl, int &ku){
    if (m == n){
        kl = max(n-2, 1);
        ku = kl;
    }else if( m > n){
        kl = n;
        ku = max(n-2 ,1);
    }else{
        ku = m;
        kl = max(m-2, 1);
    }
}
float compute_pri_res(workspace *workspace){
    float pri_res;
    int m = workspace->m;
    int n = workspace->n;
    int kl, ku;
    compute_kl_ku(m, n, kl, ku);

    cuda_mat_mul(workspace, workspace->A, workspace->rhs, workspace->Ax, m, 1, n, n, 1, m);
    cuda_vec_add_scaled(workspace, workspace->rhs_prev+n ,workspace->Ax, workspace->rhs+n,  1, -1, m);
    pri_res = cuda_vec_norm_inf(workspace, workspace->rhs_prev+n, m);

    return pri_res;
}

float compute_pri_tol(workspace *workspace){
    float max_rel_eps, temp_rel_eps;
    int m = workspace->m, n = workspace->n;
    max_rel_eps = cuda_vec_norm_inf(workspace, workspace->z, m);
    temp_rel_eps = cuda_vec_norm_inf(workspace, workspace->Ax, m);
    max_rel_eps = max(max_rel_eps, temp_rel_eps);
    return max_rel_eps;
}

float compute_dua_res(workspace *workspace){
    //NB: Use x_prev as temporary vector
    int n = workspace->n, m = workspace->m;
    int k = max(n-2 , 1);
    float dua_res;
    float *tmp;
    checkCudaErrors(cudaMalloc((void**)&tmp, sizeof(float) * n));
    checkCudaErrors(cudaMemcpy(workspace->rhs_prev, workspace->q, sizeof(float)*n, cudaMemcpyDeviceToDevice));
    cuda_mat_mul(workspace, workspace->P, workspace->x, workspace->rhs_prev, n, 1, n, n, 1, n, 1.0, 1.0);
    cuda_mat_mul(workspace, workspace->At, workspace->y, workspace->rhs_prev, n, 1, m, m, 1, n, 1.0, 1.0);
    dua_res = cuda_vec_norm_inf(workspace, workspace->rhs, n);
    return dua_res;
}
void update_info(workspace *workspace){
    int m = workspace->m;
    if(workspace->compute_objective){
        workspace->obj_val = compute_obj_val(workspace);
    }
    if(m==0){
        workspace->pri_res = 0.0;
    }else{
        workspace->pri_res = compute_pri_res(workspace);
    }
    workspace->dua_res = compute_dua_res(workspace);
}
bool check_termination(workspace *workspace){
    return workspace->pri_res < workspace->eps_pri_limit and workspace->dua_res < workspace->eps_dual_limit;
}

void admm(workspace *workspace){

}
//float comupute_dua_tol(workspace* workspace){
//    float max_rel_eps, temp_rel_eps;
//
//}

int main(){
    int m = 3, n =2;
    workspace work;
    workspace *workspace = &work;
    workspace->m = m;
    workspace->n = n;
    workspace_init(workspace);
    workspace_default_setting(workspace);
    std::vector<float>P{1.0000, 0.353553, 0.353553, 1.00000};
    std::vector<float>A{0.706628, 0.999323, 0.998647, 0.0, 0.0,0.999323};
    std::vector<float>q{0.5, 0.707106769};
    std::vector<float>upper{1.413256, 1.398106, 0.989279};
    std::vector<float>lower{1.413256, 0.0, 0.0};
    matrix_update(workspace, P, A, q, upper, lower);

    update_xztilde(workspace);
    update_x(workspace);
    update_z(workspace);
    update_y(workspace);
    checkCudaErrors(cudaMemcpy(workspace->rhs, workspace->x, sizeof(float) * n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(workspace->rhs+n, workspace->z, sizeof(float) * m, cudaMemcpyDeviceToDevice));
    update_info(workspace);
    printf("primal residual is %f, dual residual is %f\n", workspace->pri_res, workspace->dua_res);
//    print_matrix("y", workspace->y, m);
    std::swap(workspace->rhs, workspace->rhs_prev);
    print_matrix("rhs", workspace->rhs, 1, m+n);
    return 0;
}

