#include <time.h>
#include <iostream>
#include <vector>
#include <emmintrin.h>
#include <immintrin.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>


/*
@input
mat_x: m x k size matrix
mat_y: k x n size matrix

@output
mat_z: m x n size matrix 
*/

void matmult_baseline(int M, int N, int K, const float* mat_x, const float* mat_y, float* mat_z)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < K; k++)
				mat_z[N * i + j] += mat_x[i * K + k] * mat_y[k * N + j];
}

void matmult_kij(int M, int N, int K, const float* mat_x, const float* mat_y, float* mat_z)
{
	float tmp;
	for (int k = 0; k < K; k++)
	{
		for (int i = 0; i < M; i++)
		{
			tmp = mat_x[i * N + k];
			for (int j = 0; j < N; j++)
				mat_z[N * i + j] += tmp * mat_y[k * K + j];
		}
	}
}

void matmul_SIMD(int M, int N, int K, float* mat_x, float* mat_y, float* mat_z)
{
	for (int i = 0; i < M; ++i)
	{
		for (int j = 0; j < N; j += 4)
		{
			__m128 vR = _mm_setzero_ps();
			for (int k = 0; k < K; k++)
			{
				__m128 vA = _mm_set1_ps(mat_x[M * i + K]);
				__m128 vB = _mm_loadu_ps(&mat_y[k * K + j]);
				vR = _mm_add_ps(vR, _mm_mul_ps(vA, vB));
			}
			_mm_storeu_ps(&mat_z[i * M + j], vR);
		}
	}
}

void matmul_SIMD_kij(int M, int N, int K, float* mat_x, float* mat_y, float* mat_z)
{
	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			__m128 vA = _mm_set1_ps(mat_x[i * M + k]);
			for (int j = 0; j < N; j += 4)
			{
				__m128 vB = _mm_loadu_ps(&mat_y[k * K + j]);
				__m128 vR = _mm_loadu_ps(&mat_z[i * M + j]);
				vR = _mm_add_ps(vR, _mm_mul_ps(vA, vB));
				_mm_storeu_ps(&mat_z[i * M + j], vR);
			}
		}
	}
}

void randomInitMatrix(int N, int M, std::vector<float>& mat)
{
	srand(time(0));
	mat.resize(N * M);
	for (int i = 0; i < mat.size(); i++) mat[i] = rand() % 100;
}

void zeroInitMatrix(int N, int M, std::vector<float>& mat)
{
	srand(time(0));
	mat.resize(N * M);
	for (int i = 0; i < mat.size(); i++) mat[i] = 0;
}

void InitMatrix(int N,int M,int K,std::vector<float>& mat_x, std::vector<float>& mat_y, std::vector<float>& mat_z)
{
	randomInitMatrix(M, K, mat_x);
	randomInitMatrix(K, N, mat_y);
	zeroInitMatrix(M, N, mat_z);
}

void randomInitOpencvMatrix(int n, int m, cv::Mat& mat)
{
	srand(time(0));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			mat.at<float>(i, j) = rand() % 100;
}


int main()
{
	int N = 1000;
	int M = 1000;
	int K = 1000;
	const int iteration = 10;
	bool useOpencvMatMul = false;

	std::vector<float> mat_x;
	std::vector<float> mat_y;
	std::vector<float> mat_z;
	clock_t start, end;


	InitMatrix(N, M, K, mat_x, mat_y, mat_z);
	double elapsed = 0;

	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		matmult_baseline(N, M, K, &mat_x[0], &mat_y[0], &mat_z[0]);
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("Baseline : %lf ms\n", elapsed / iteration);


	InitMatrix(N, M, K, mat_x, mat_y, mat_z);
	elapsed = 0;
	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		matmult_kij(N, M, K, &mat_x[0], &mat_y[0], &mat_z[0]);
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("kij : %lf ms\n", elapsed / iteration);

	InitMatrix(N, M, K, mat_x, mat_y, mat_z);
	elapsed = 0;
	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		matmul_SIMD(N, M, K, &mat_x[0], &mat_y[0], &mat_z[0]);
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("SIMD : %lf ms\n", elapsed / iteration);

	InitMatrix(N, M, K, mat_x, mat_y, mat_z);
	elapsed = 0;
	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		matmul_SIMD_kij(N, M, K, &mat_x[0], &mat_y[0], &mat_z[0]);
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("SIMD_kij : %lf ms\n", elapsed / iteration);


	// ================ OpenCV ================ // 
	cv::Mat cv_mat_x(M, K, CV_32FC1);
	cv::Mat cv_mat_y(K, N, CV_32FC1);
	randomInitOpencvMatrix(M, K, cv_mat_x);
	randomInitOpencvMatrix(K, N, cv_mat_y);
	cv::Mat cv_mat_z(M, N, CV_32FC1, cv::Scalar(0.0));

	elapsed = 0;
	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		cv_mat_z = cv_mat_x * cv_mat_y;
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("OpenCV : %lf ms\n", elapsed / iteration);


	

	// ================ Eigen ================ // 

	Eigen::MatrixXd eigen_mat_x = Eigen::MatrixXd(M, K);
	Eigen::MatrixXd eigen_mat_y = Eigen::MatrixXd(K, N);
	Eigen::MatrixXd eigen_mat_z = Eigen::MatrixXd(M, N);

	srand(time(0));
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			eigen_mat_x(i,j) = rand() % 100;

	for (int i = 0; i < K; i++)
		for (int j = 0; j < N; j++)
			eigen_mat_y(i, j) = rand() % 100;

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			eigen_mat_z(i, j) = 0;

	elapsed = 0;
	for (int i = 0; i < iteration; i++)
	{
		start = clock();
		eigen_mat_z = eigen_mat_x * eigen_mat_y;
		end = clock();
		elapsed += (double)(end - start);;
	}
	printf("Eigen : %lf ms\n", elapsed / iteration);


	return 0;
}


