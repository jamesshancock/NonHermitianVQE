#ifndef GRADS_H
#define GRADS_H
#include <Eigen/Dense>
#include <complex>
#include <cmath>

// System parameters
#define sysMATRIX Eigen::Matrix<std::complex<double>, N, N>
#define sysVECTOR Eigen::Matrix<std::complex<double>, N, 1>

// Random useful defines
#define gateMATRIX Eigen::Matrix<std::complex<double>, 2, 2>
#define varyMATRIX Eigen::MatrixXcd

sysMATRIX Hamiltonian(const float Bz, const float b) {
    sysMATRIX H;
    std::complex<double> ib(0, b);
    sysMATRIX Matrix;

    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = Z;
        Term[(i + 1) % n] = Z;
        Matrix = termMaker(Term);
        H += -Matrix;
    }
    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = Z;
        Matrix = termMaker(Term);
        H += -Bz * Matrix;
    }
    for (int i = 0; i < n; i++) {
        std::vector<gateMATRIX> Term(n, I);
        Term[i] = X;
        Matrix = termMaker(Term);
        H += -ib * Matrix;
    }
    return H;
}

sysMATRIX MTIM(float Bz, float b, float ER, float EI) {
    sysMATRIX M, H, III;

    std::vector<gateMATRIX> identityTerm(n, I);
    III = termMaker(identityTerm);

    H = Hamiltonian(Bz, b);

    std::complex<double> E(ER, EI);
    M = (H.adjoint() - std::conj(E) * III) * (H - E * III);

    return M;
}

sysVECTOR VectorGrad(const sysMATRIX& Matrix, const sysVECTOR& Vector, const float stepSize) {
    sysVECTOR grad;
    grad.setZero();
    for (int i = 0; i < 8; ++i) {
        Vector(i) += stepSize;
        float result1 = innerProduct(Matrix, Vector);
        Vector(i) -= 2 * stepSize;
        float result2 = innerProduct(Matrix, Vector);
        Vector(i) += stepSize;
        grad(i) = (result1 - result2) / (2 * stepSize);
    }
    return grad;
}

float ERGrad(const float Bz, const float b, const float ER, const float EI,
    const sysVECTOR& Vector, const float stepSize) {
    sysMATRIX M1 = MTIM(Bz, b, ER + stepSize, EI);
    sysMATRIX M2 = MTIM(Bz, b, ER - stepSize, EI);
    float cost1 = innerProduct(M1, Vector);
    float cost2 = innerProduct(M2, Vector);
    return (cost1 - cost2) / (2 * stepSize);
}

float EIGrad(const float Bz, const float b, const float ER, const float EI,
    const sysVECTOR& Vector, const float stepSize) {
    sysMATRIX M1 = MTIM(Bz, b, ER, EI + stepSize);
    sysMATRIX M2 = MTIM(Bz, b, ER, EI - stepSize);
    float cost1 = innerProduct(M1, Vector);
    float cost2 = innerProduct(M2, Vector);
    return (cost1 - cost2) / (2 * stepSize);
}

float innerProduct(const sysMATRIX& Matrix, const sysVECTOR& Vector) {
    std::complex<double> mv, result = 0.0, norm = 0.0;
    for (int i = 0; i < 8; ++i) {
        norm += std::conj(Vector(i)) * Vector(i);
        mv = 0.0;
        for (int j = 0; j < 8; ++j) {
            mv += Matrix(i, j) * Vector(j);
        }
        result += std::conj(Vector(i)) * mv;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-noiseLevel, noiseLevel);
    double noise = dis(gen);
    result /= norm;
    result += noise;
    return result.real();
}
#endif