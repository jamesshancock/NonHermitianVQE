#ifndef KRONECKERS_H
#define KRONECKERS_H

#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

#define sysMATRIX Eigen::Matrix<std::complex<double>, N, N>
#define sysVECTOR Eigen::Matrix<std::complex<double>, N, 1>

// Random useful defines
#define gateMATRIX Eigen::Matrix<std::complex<double>, 2, 2>
#define varyMATRIX Eigen::MatrixXcd
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> DynamicMatrix;

// Kronecker product function
sysMATRIX tensorProduct(const std::vector<gateMATRIX>& matrices) {
    if (matrices.empty()) {
        return DynamicMatrix();
    }

    // Start with the last matrix instead of the first
    DynamicMatrix result = matrices.back();

    // Iteratively take the Kronecker product of the result with the previous matrix
    // Note: We start from the second-to-last element and move backwards
    for (int i = matrices.size() - 2; i >= 0; --i) {
        DynamicMatrix temp = DynamicMatrix::Zero(result.rows() * 2, result.cols() * 2);

        for (int j = 0; j < result.rows(); ++j) {
            for (int k = 0; k < result.cols(); ++k) {
                temp.block(j * 2, k * 2, 2, 2) = result(j, k) * matrices[i];
            }
        }

        result = temp;
    }

    return result;
}

// Hamiltonians

sysMATRIX Hamiltonian(const float Bz, const float b) {
    /*
    * This creates the Hamiltonian for the transverse Ising model
    *
    * Parameters:
    * Bz: The transverse field strength
    * b: The coupling strength
    *
    * Returns:
    * H: The Hamiltonian
    */
    sysMATRIX H = sysMATRIX::Zero(); // Initialize H to zero
    std::complex<double> ib(0.0, b); // Initialize complex number
    sysMATRIX Matrix;

    // Add ZZ interactions
    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = Z;
        Term[(i + 1) % n] = Z;
        Matrix = tensorProduct(Term);
        H += -Matrix;
    }

    // Add Bz * Z interactions
    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = Z;
        Matrix = tensorProduct(Term);
        H += -Bz * Matrix;
    }

    // Add ib * X interactions
    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = X;
        Matrix = tensorProduct(Term);
        H += -ib * Matrix;
    }

    return H;
}

sysMATRIX HamiltonianQHO(const float omega, const float lambda) {
    /*
    * This creates the Hamiltonian for the quantum harmonic oscillator
    *
    * Parameters:
    * omega: The frequency of the oscillator
    * lambda: The coupling strength
    *
    * Returns:
    * H: The Hamiltonian
    */
    sysMATRIX H = sysMATRIX::Zero(); // Initialize H to zero
    sysMATRIX Matrix;
    sysMATRIX zed;
    std::vector<gateMATRIX> IdentityTerm(n, I);
    sysMATRIX Identity = tensorProduct(IdentityTerm);
    std::complex<double> ilambda(0, lambda / sqrt(2));

    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I); // Reset Term to identity matrices
        Term[i] = Z;
        zed = tensorProduct(Term);
        Matrix = Identity - 1.0 / 2.0 * zed; // Ensure proper division
        H += omega * Matrix;
    }

    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = X;
        for (int j = 0; j < i; j++) {
            Term[j] = Z;
        }
        Matrix = tensorProduct(Term);
        H += ilambda * Matrix;
    }

    return H;
}

sysMATRIX MTIM(float Bz, float b, float ER, float EI) {
    /*
    * This creates the matrix M = (H^dagger - E~*I)(H - E*I)
    *
    * Parameters:
    * Bz: The transverse field strength
    * b: The coupling strength
    * ER: The real part of the eigenvalue
    *
    * Returns:
    * M: The matrix M
    */
    sysMATRIX M, H, III;

    std::vector<gateMATRIX> identityTerm(n, I);
    III = tensorProduct(identityTerm);

    H = Hamiltonian(Bz, b);

    std::complex<double> E(ER, EI);
    M = (H.adjoint() - std::conj(E) * III) * (H - E * III);
    return M;
}

sysMATRIX MQHO(float omega, float lambda, float ER, float EI) {
    /*
    * This creates the matrix M = (H^dagger - E~*I)(H - E*I)
    *
    * Parameters:
    * Bz: The transverse field strength
    * b: The coupling strength
    * ER: The real part of the eigenvalue
    *
    * Returns:
    * M: The matrix M
    */
    sysMATRIX M, H, III;

    std::vector<gateMATRIX> identityTerm(n, I);
    III = tensorProduct(identityTerm);

    H = HamiltonianQHO(omega, lambda);

    std::complex<double> E(ER, EI);
    M = (H.adjoint() - std::conj(E) * III) * (H - E * III);

    return M;
}

// Random useful functions
float innerProduct(const sysMATRIX& Matrix, const sysVECTOR& Vector) {
    /*
    * This computes the inner product of a vector with a matrix
    *
    * Parameters:
    * Matrix: The matrix
    * Vector: The vector
    *
    * Returns:
    * result: The inner product
    */
    std::complex<double> mv, result = 0.0, norm = 0.0;
    for (int i = 0; i < N; ++i) {
        norm += std::conj(Vector(i)) * Vector(i);
        mv = 0.0;
        for (int j = 0; j < N; ++j) {
            mv += Matrix(i, j) * Vector(j);
        }
        result += std::conj(Vector(i)) * mv;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-noiseLevel, noiseLevel);
    double noise = dis(gen);
    if (norm != 0.0) {
        result /= norm;
    }
    result += noise;
    return result.real();
}

float innerProductNL(const sysMATRIX& Matrix, const sysVECTOR& Vector) {
    /*
    * This computes the inner product of a vector with a matrix
    *
    * Parameters:
    * Matrix: The matrix
    * Vector: The vector
    *
    * Returns:
    * result: The inner product
    */
    std::complex<double> mv, result = 0.0, norm = 0.0;
    for (int i = 0; i < 8; ++i) {
        norm += std::conj(Vector(i)) * Vector(i);
        mv = 0.0;
        for (int j = 0; j < 8; ++j) {
            mv += Matrix(i, j) * Vector(j);
        }
        result += std::conj(Vector(i)) * mv;
    }
    result /= norm;
    return result.real();
}

std::vector<float> linspace(float start, float end, int n) {
    std::vector<float> result;
    float step = (end - start) / (n - 1);

    for (int i = 0; i < n; i++) {
        result.push_back(start + i * step);
    }

    return result;
}

float meanf(const std::vector<float>& v) {
    float sum = std::accumulate(v.begin(), v.end(), 0.0f);
    float mean = sum / v.size();
    return mean;
}

float stdevf(const std::vector<float>& v) {
	float sum = std::accumulate(v.begin(), v.end(), 0.0f);
	float mean = sum / v.size();

	float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0f);
	float stdev = std::sqrt(sq_sum / v.size() - mean * mean);

	return stdev;
}



#endif