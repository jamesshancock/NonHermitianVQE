/*
 Made by James
*/

// == Includes ==
#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <functional>

// == System parameters ==
// n is the number of qubits
// N is the size of the Hilbert space
constexpr int n = 5;

constexpr int N = 1 << n; 

// sysMATRIX is the size of the Hilbert space matries
// sysVECTOR is the size of the Hilbert space vectors
#define sysMATRIX Eigen::Matrix<std::complex<double>, N, N>  
#define sysVECTOR Eigen::Matrix<std::complex<double>, N, 1>

// Random useful defines
// gateMATRIX is the size of the single qubit matrices
#define gateMATRIX Eigen::Matrix<std::complex<double>, 2, 2>
#define varyMATRIX Eigen::MatrixXcd
const gateMATRIX I = gateMATRIX::Identity();
const gateMATRIX X = (gateMATRIX() << 0, 1, 1, 0).finished();
const gateMATRIX Z = (gateMATRIX() << 1, 0, 0, -1).finished();

// == Test parameters ==
const std::string System = "CTIM"; // Choose between "CTIM" and "CQHO"
const std::string Observ = "Xmag"; // Choose between "Rmin", "Imax" , "Xmag", "Sus"
const std::string exactOrQuantum = "Exact"; // Choose between "exact" and "quantum"
const float noiseLevel = 4e-20;
const float variableMin = 0.0; // }
const float variableMax = 3.5; // } b or lambda
const int variablePoints = 30; // }
const float Constant = 0.0; // Bz or omega
const int firstStepIters = 400;
const int secondStepIters = 1000;
const int TOTALRUNS = 1;
const int maxLength = n; // Number of samples considered (For average-based stopping criteria)
const float avTol = 0.1; // Difference between average and current value (For average-based stopping criteria)
float costTol = 0.1; // Cost value that must be reached before starting the average-based stopping criteria

// == Modules ==
#include "kroneckers.h"

// == Gradient descent parameters ==
const float stepSize = 0.1;
const float gamma = 0.1; 

// == Adam parameters ==
constexpr float alpha = 0.01; // Learning rate
constexpr float beta1 = 0.9;
constexpr float beta2 = 0.999;
constexpr float epsilon = 1e-8;

sysVECTOR VectorGrad(const sysMATRIX& Matrix, sysVECTOR& Vector, float stepSize) {
    /*
    * This computes the gradient of the inner product of a vector with a matrix
    *
    * Parameters:
    * Matrix: The matrix
    * Vector: The vector
    * stepSize: The step size for the finite difference
    *
    * Returns:
    * grad: The gradient
    */
    sysVECTOR grad;
    grad.setZero();
    for (int i = 0; i < N; ++i) {
        std::complex<double> originalValue = Vector(i);

        // Perturb the real part
        Vector(i) = originalValue + std::complex<double>(stepSize, 0);
        float resultReal1 = innerProduct(Matrix, Vector);
        Vector(i) = originalValue - std::complex<double>(stepSize, 0);
        float resultReal2 = innerProduct(Matrix, Vector);

        // Perturb the imaginary part
        Vector(i) = originalValue + std::complex<double>(0, stepSize);
        float resultImag1 = innerProduct(Matrix, Vector);
        Vector(i) = originalValue - std::complex<double>(0, stepSize);
        float resultImag2 = innerProduct(Matrix, Vector);

        Vector(i) = originalValue; // Reset to original value

        // Compute gradient components
        double gradReal = (resultReal1 - resultReal2) / (2 * stepSize);
        double gradImag = (resultImag1 - resultImag2) / (2 * stepSize);

        grad(i) = std::complex<double>(gradReal, gradImag);
    }
    return grad;
}

sysVECTOR VectorGradReal(const sysMATRIX& Matrix, sysVECTOR& Vector, float stepSize) {
    sysVECTOR grad;
    grad.setZero();
    for (int i = 0; i < N; ++i) {
        std::complex<double> originalValue = Vector(i);

        // Perturb the real part
        Vector(i) = originalValue + std::complex<double>(stepSize, 0);
        float resultReal1 = innerProduct(Matrix, Vector);
        Vector(i) = originalValue - std::complex<double>(stepSize, 0);
        float resultReal2 = innerProduct(Matrix, Vector);

        Vector(i) = originalValue; // Reset to original value

        // Check for NaN values
        if (std::isnan(resultReal1) || std::isnan(resultReal2)) {
            //std::cerr << "NaN detected in innerProduct results: resultReal1 = " << resultReal1 << ", resultReal2 = " << resultReal2 << std::endl;
			resultReal1 = 0.0;
        }

        // Compute gradient component for the real part
        double gradReal = (resultReal1 - resultReal2) / (2 * stepSize);

        // Check for NaN in gradient
        if (std::isnan(gradReal)) {
            //std::cerr << "NaN detected in gradient calculation: gradReal = " << gradReal << std::endl;
			gradReal = 0.0;
        }

        grad(i) = std::complex<double>(gradReal, 0.0); // Only real part
    }
    return grad;
}

float XMagnet(const sysVECTOR& Vector) {
    /*
    * This computes the magnetization in the x direction
    * 
    * Parameters:
    * Vector: The vector
    * 
    * Returns:
    * Xmag: The magnetization in the x direction
	*/
    auto Xtotal = std::make_unique<sysMATRIX>();
    Xtotal->setZero();
    std::vector<gateMATRIX> Term(n, I);
    for (int i = 0; i < n; i++) {
        Term[i] = X;
        *Xtotal += tensorProduct(Term); // Assume tensorProduct is optimized for pointers or references
        Term[i] = I;
    }
    return abs(innerProduct(*Xtotal, Vector)) / n;
}

float XSus(const sysVECTOR& Vector) {
	/*
	* This computes the susceptibility in the x direction
	*
	* Parameters:
	* Vector: The vector
	*
	* Returns:
	* XSus: The susceptibility in the x direction
	*/
    auto Xtotal = std::make_unique<sysMATRIX>();
    Xtotal->setZero();
    std::vector<gateMATRIX> Term(n, I);
    for (int i = 0; i < n; i++) {
        Term[i] = X;
        *Xtotal += tensorProduct(Term); // Assume tensorProduct is optimized for pointers or references
        Term[i] = I;
    }
	auto X2total = std::make_unique<sysMATRIX>(*Xtotal * *Xtotal);
	float term1 = abs(innerProduct(*Xtotal, Vector));
	float term2 = abs(innerProduct(*X2total, Vector));
    return (term2 - term1 * term1)/n;
}

float ERGrad(const float Bz, const float b, const float ER, const float EI,
             const sysVECTOR& Vector, const float stepSize, std::string System) {
    std::unique_ptr<sysMATRIX> M1, M2;
    float cost1, cost2, grad;

    if (System == "CTIM") {
        M1 = std::make_unique<sysMATRIX>(MTIM(Bz, b, ER + stepSize, EI));
        M2 = std::make_unique<sysMATRIX>(MTIM(Bz, b, ER - stepSize, EI));
    }
    else if (System == "CQHO") {
        M1 = std::make_unique<sysMATRIX>(MQHO(Bz, b, ER + stepSize, EI));
        M2 = std::make_unique<sysMATRIX>(MQHO(Bz, b, ER - stepSize, EI));
    }

    cost1 = innerProduct(*M1, Vector);
    cost2 = innerProduct(*M2, Vector);
    grad = (cost1 - cost2) / (2 * stepSize);
    return grad;
}

float EIGrad(const float Bz, const float b, const float ER, const float EI,
    const sysVECTOR& Vector, const float stepSize, std::string System) {
    std::unique_ptr<sysMATRIX> M1, M2;
    float cost1, cost2, grad;

    if (System == "CTIM") {
        M1 = std::make_unique<sysMATRIX>(MTIM(Bz, b, ER, EI + stepSize));
        M2 = std::make_unique<sysMATRIX>(MTIM(Bz, b, ER, EI - stepSize));
    }
    else if (System == "CQHO") {
        M1 = std::make_unique<sysMATRIX>(MQHO(Bz, b, ER, EI + stepSize));
        M2 = std::make_unique<sysMATRIX>(MQHO(Bz, b, ER, EI - stepSize));
    }

    cost1 = innerProduct(*M1, Vector);
    cost2 = innerProduct(*M2, Vector);
    grad = (cost1 - cost2) / (2 * stepSize);
    return grad;
}

std::tuple<sysVECTOR, std::complex<double>, int> eigenfinder(const float omega, const float lambda, const float stepSize, const std::string& Type,
                                                                              float gamma, const int firstPartIters, const int secondPartIters, std::string System) {
    /*
    * This function finds the groundstate and spectral eigenvalues and eigenvectors of the Hamiltonian
    * 
    * Parameters:
    * omega: The transverse field strength
    * lambda: The coupling strength
    * stepSize: The step size for the finite difference
    * Type: The type of eigenvalue to find
    * gamma: The learning rate
    * firstPartIters: The number of iterations for the first part
    * secondPartIters: The number of iterations for the second part
    * System: The system to use
    * 
    * Returns:
    * Vector: The eigenvector
    * E: The eigenvalue
    * counter: The number of iterations
    */
    float Rmin, Imax, ER, EI;
    int counter = 0;
    int totalIters = firstPartIters + secondPartIters;
    std::vector<float> lastFiveCosts;
    auto Vector = std::make_unique<sysVECTOR>();
    std::function<sysMATRIX(float, float, float, float)> Mfunction;
    std::function<float(float, float, float, float, const sysVECTOR&, float)> EIfunction, ERfunction;
    auto M = std::make_unique<sysMATRIX>();
    std::vector<std::string> sequence, firstPartSequence, seq;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::complex<double> Gamma(gamma, 0);
    if (System == "CTIM") {
        Rmin = -1.5 * n -lambda * n;
		Imax = n/2 * lambda; // Non-Hermitian
        //Imax = 0; // Hermitian
        Mfunction = MTIM;
    }
    else if (System == "CQHO") {
        Rmin = (0.5 * n);
        Imax = n * lambda;
        Mfunction = MQHO;
    }

    bool STOP = false;
    if (Type == "groundstate") {
        ER = Rmin;
		sequence = { "para", "imag", "real" }; // Non-Hermitian
        //sequence = { "para", "real" }; // Hermitian
        EI = Imax;
        Vector->setZero();
		for (int i = 0; i < N; ++i) {
			std::complex<double> val = std::complex<double>(std::normal_distribution<float>(0.0, 1.0)(gen), std::normal_distribution<float>(0.0, 1.0)(gen));
			Vector->operator()(i) = val;
		}
		*Vector /= Vector->norm();
		
    }
    else if (Type == "spectral") {
        EI = Imax;
        sequence = { "para", "real", "imag" };
        ER = abs(Rmin);
        Vector->setOnes();
        *Vector *= std::complex<double>(1.0, 1.0);
    }

    float inputGamma = gamma;
	//firstPartSequence = { sequence[0], sequence[1] }; // Non-Hermitian
	firstPartSequence = { sequence[0] }; // Hermitian
    int diff = 0;
    while (STOP == false) {
        *M = Mfunction(omega, lambda, ER, EI);
        seq = (counter < firstPartIters) ? firstPartSequence : sequence;
        for (const auto& typ : seq) {
            if (typ == "para") {
				auto grad = std::make_unique<sysVECTOR>(VectorGrad(Mfunction(omega, lambda, ER, EI), *Vector, stepSize)); // Non-Hermitian
				//auto grad = std::make_unique<sysVECTOR>(VectorGradReal(Mfunction(omega, lambda, ER, EI), *Vector, stepSize)); // Hermitian
                *Vector -= Gamma * (*grad);
            }
            else if (typ == "imag") {
                float grad = EIGrad(omega, lambda, ER, EI, *Vector, stepSize, System);
                EI -= gamma * grad;
            }
            else if (typ == "real") {
                float grad = ERGrad(omega, lambda, ER, EI, *Vector, stepSize, System);
                ER -= gamma * grad;
            }
        }
        counter++;
        float cost = abs(innerProduct(*M, *Vector));
		bool PRINT = false;
        if (PRINT == true and counter % 30 == 0) {
            std::cout << "-------" << std::endl;
            std::cout << "ER: " << ER << std::endl;
            std::cout << "EI: " << EI << std::endl;
            std::cout << "Cost: " << cost << std::endl;
        }
        
        lastFiveCosts.push_back(cost);
        //if (cost < 1.0) {
        //    gamma = inputGamma * cost;
        //    Gamma = std::complex<double>(gamma, 0);
        //}
        if (counter >= totalIters) {
            std::cout << "Ended due to max iters" << std::endl;
            STOP = true;
        }
        if (cost < costTol) {
            if (lastFiveCosts.size() > maxLength) {
                lastFiveCosts.erase(lastFiveCosts.begin());
                float sum = 0.0;
				
                // Print the last five costs
                //std::cout << "Last five costs: ";
                for (const auto& cost : lastFiveCosts) {
                    //std::cout << cost << " ";
                    sum += cost;
                }
                float av = sum / lastFiveCosts.size();
                if (abs(cost - av) < avTol and counter < firstPartIters) {
                    std::cout << "First part iters: " << counter << std::endl;
                    diff = firstPartIters - counter;
                    counter = firstPartIters;
                    lastFiveCosts.clear();
                    av = 0.0;
                }

                if (abs(cost - av) < avTol) {
                    std::cout << "Ended due to average tol" << std::endl;
                    STOP = true;
                }
            }

		}
    }
    float cost = abs(innerProduct(*M, *Vector));
    counter -= diff;
    std::complex<double> E = std::complex<double>(ER, EI);
    std::cout << "Final cost: " << cost << std::endl;
    return std::make_tuple(*Vector, E, counter);
}



// == Solver ==
std::tuple<sysVECTOR, std::complex<double>, int, float> solverf(const float b, const float stepSize,
                                                                          const float gamma, const int firstPartIters,
                                                                          const int secondPartIters, const std::string Solver,
                                                                          const std::string System) {
    /*
    * This function runs the solver for a given coupling strength
    * 
    * Parameters:
    * b: The coupling strength
    * stepSize: The step size for the finite difference
    * gamma: The learning rate
    * firstPartIters: The number of iterations for the first part
    * secondPartIters: The number of iterations for the second part
    * Solver: The solver to use
    * System: The system to use
    * 
    * Returns:
    * Vector1: The eigenvector
    * E1: The eigenvalue
    * iterations: The number of iterations
	*/
    float Bz = Constant; // This also acts as omega for the QHO
    auto Vector1 = std::make_unique<sysVECTOR>();
	auto GSVector1 = std::make_unique<sysVECTOR>();
    std::complex<double> E1;
    int iterations;
    std::vector<std::unique_ptr<sysVECTOR>> VectorList;
    std::vector<std::complex<double>> EList;
    float minRealPart, maxImagPart;
    auto H = std::make_unique<sysMATRIX>();
    if (System == "CTIM") {
        *H = Hamiltonian(Bz, b);
    }
    else if (System == "CQHO") {
        *H = HamiltonianQHO(Bz, b);
    }

    // Solve for eigenvalues and eigenvectors
    auto solver = Eigen::ComplexEigenSolver<sysMATRIX>(*H);
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    std::complex<double> groundstateE = eigenvalues[0];
    std::complex<double> spectralE = eigenvalues[0];

    *Vector1 = eigenvectors.col(0);
	*GSVector1 = eigenvectors.col(0);
    minRealPart = groundstateE.real();
    maxImagPart = spectralE.imag();

    // Iterate through eigenvalues to find the one with the lowest real part
    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double realPart = eig.real();
        if (realPart < minRealPart) {
            minRealPart = realPart;
            groundstateE = eig;
            *GSVector1 = eigenvectors.col(i); // Get the corresponding eigenvector
			*GSVector1 /= GSVector1->norm();
        }
    }

    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double imagPart = eig.imag();
        if (imagPart > maxImagPart) {
            maxImagPart = imagPart;
            spectralE = eig;
            *Vector1 = eigenvectors.col(i); // Get the corresponding eigenvector 
        }
    }
   
    // Now that you have the eigenvector corresponding to the lowest real part eigenvalue,
    // you can compute XMagnet
	float xMagnetValue = 0.0;
    if (Observ == "Xmag") {
        xMagnetValue = XMagnet(*GSVector1);
    }
    else if (Observ == "Sus") {
		xMagnetValue = XSus(*GSVector1);
    }    
    std::tuple<sysVECTOR, std::complex<double>, int> result1;
    if (Observ == "Rmin" or Observ == "Xmag" or Observ == "Sus") {
        std::cout << "Groundstate energy: " << groundstateE << std::endl;
		if (System == "CTIM") {
			auto M = std::make_unique<sysMATRIX>(MTIM(Bz, b, groundstateE.real(), groundstateE.imag()));
			std::cout << "Ideal cost: " << abs(innerProduct(*M, *GSVector1)) << std::endl;

		}
		else if (System == "CQHO") {
			auto M = std::make_unique<sysMATRIX>(MQHO(Bz, b, groundstateE.real(), groundstateE.imag()));
			std::cout << "Ideal cost: " << abs(innerProduct(*M, *GSVector1)) << std::endl;
		}
        std::cout << Observ <<  " of groundstate: " << xMagnetValue << std::endl;
        result1 = eigenfinder(Bz, b, stepSize, "groundstate", gamma, firstStepIters, secondStepIters, System);
        *Vector1 = std::get<0>(result1);
        E1 = std::get<1>(result1);
        iterations = std::get<2>(result1);
        std::cout << "Estimate groundstate eigenvalue: " << E1 << std::endl;
    }
    else if (Observ == "Imax") {
        std::cout << "Spectral energy: " << spectralE << std::endl;
        result1 = eigenfinder(Bz, b, stepSize, "spectral", gamma, firstStepIters, secondStepIters, System);
        *Vector1 = std::get<0>(result1);
        E1 = std::get<1>(result1);
        iterations = std::get<2>(result1);
        std::cout << "Estimate spectral eigenvalue: " << E1 << std::endl;
    }

    return std::make_tuple(*Vector1, E1, iterations, xMagnetValue);
}


std::tuple<std::vector<float>, std::vector<int>, std::vector<float>> runSolverXmag(const std::vector<float> bList, const float stepSize, const float gamma, 
                                 const int firstPartIters, const int secondPartIters, const std::string Solver, 
                                 const std::string System) {
    /*
    * This function runs the solver for a list of coupling strengths and computes the magnetization in the x direction
    * 
    * Parameters:
    * bList: The list of coupling strengths
    * stepSize: The step size for the finite difference
    * gamma: The learning rate
    * firstPartIters: The number of iterations for the first part
    * secondPartIters: The number of iterations for the second part
    * 
    * Returns:
    * XMag: The magnetization in the x direction
    */
    std::vector<sysVECTOR> Vectors;
    std::vector<std::complex<double>> Es;
    std::tuple<sysVECTOR, std::complex<double>, int, float> results;
    sysVECTOR gsVector, sVector;
    std::vector<float> XMag;
    std::vector<int> Iterations;
	std::vector<float> TrueXMag;
    for (const auto& b : bList) {
        std::cout << "=============" << std::endl;
        std::cout << "Variable: " << b << std::endl;
        results = solverf(b, stepSize, gamma, firstStepIters, secondStepIters, Solver, System);
        gsVector = std::get<0>(results);
        float Xmag = XMagnet(gsVector);
        XMag.push_back(Xmag);
        Iterations.push_back(std::get<2>(results));
		TrueXMag.push_back(std::get<3>(results));
        std::cout << "Xmag: " << Xmag << std::endl;
        std::cout << "Iterations: " << Iterations.back() << std::endl;

    }
    return std::make_tuple(XMag, Iterations, TrueXMag);
}

std::tuple<std::vector<float>, std::vector<int>, std::vector<float>> runSolverXSus(const std::vector<float> bList, const float stepSize, const float gamma,
    const int firstPartIters, const int secondPartIters, const std::string Solver,
    const std::string System) {
    /*
    * This function runs the solver for a list of coupling strengths and computes the magnetization in the x direction
    *
    * Parameters:
    * bList: The list of coupling strengths
    * stepSize: The step size for the finite difference
    * gamma: The learning rate
    * firstPartIters: The number of iterations for the first part
    * secondPartIters: The number of iterations for the second part
    *
    * Returns:
    * XSus: The suscepability in the x direction
    */
    std::vector<sysVECTOR> Vectors;
    std::vector<std::complex<double>> Es;
    std::tuple<sysVECTOR, std::complex<double>, int, float> results;
    sysVECTOR gsVector, sVector;
    std::vector<float> XSuS;
    std::vector<int> Iterations;
    std::vector<float> TrueXMag;
    for (const auto& b : bList) {
        std::cout << "=============" << std::endl;
        std::cout << "Variable: " << b << std::endl;
        results = solverf(b, stepSize, gamma, firstStepIters, secondStepIters, Solver, System);
        gsVector = std::get<0>(results);
        float Xsus = XSus(gsVector);
        XSuS.push_back(Xsus);
        Iterations.push_back(std::get<2>(results));
		TrueXMag.push_back(std::get<3>(results));
        std::cout << "XSus: " << Xsus << std::endl;
        std::cout << "Iterations: " << Iterations.back() << std::endl;

    }
    return std::make_tuple(XSuS, Iterations, TrueXMag);
}


std::tuple<std::vector<float>, std::vector<int>> runSolverRmin(const std::vector<float> bList, const float stepSize, const float gamma,
    const int firstPartIters, const int secondPartIters, const std::string Solver, const std::string System) {
    std::vector<float> Rmins;
    std::vector<int> Iterations;
    for (const auto& b : bList) {
        std::cout << "=============" << std::endl;
        std::cout << "Variable: " << b << std::endl;
        auto results = solverf(b, stepSize, gamma, firstStepIters, secondStepIters, Solver, System);

        std::complex<double> E = std::get<1>(results);
        Rmins.push_back(abs(E.imag()));
        Iterations.push_back(std::get<2>(results)); // Capture the iterations

        std::cout << "Rmin: " << Rmins.back() << std::endl;
        std::cout << "Iterations: " << Iterations.back() << std::endl;
    }
    return std::make_tuple(Rmins, Iterations);
}

std::tuple<std::vector<float>, std::vector<int>> runSolverImax(const std::vector<float> bList, const float stepSize, const float gamma,
    const int firstPartIters, const int secondPartIters, const std::string Solver, std::string System) {
    std::vector<float> Imaxs;
    std::vector<int> Iterations;
    for (const auto& b : bList) {
        std::cout << "=============" << std::endl;
        std::cout << "Variable: " << b << std::endl;
        auto results = solverf(b, stepSize, gamma, firstStepIters, secondStepIters, Solver, System);

        std::complex<double> E = std::get<1>(results);
        Imaxs.push_back(abs(E.imag()));
        Iterations.push_back(std::get<2>(results)); // Capture the iterations

        std::cout << "Imax: " << Imaxs.back() << std::endl;
        std::cout << "Iterations: " << Iterations.back() << std::endl;
    }
    return std::make_tuple(Imaxs, Iterations);
}

float exactXmag(const float b, const std::string Solver,
    const std::string System) {

    float Bz = Constant; // This also acts as omega for the QHO
    auto Vector1 = std::make_unique<sysVECTOR>();
    auto GSVector1 = std::make_unique<sysVECTOR>();
    std::complex<double> E1;
    int iterations;
    std::vector<std::unique_ptr<sysVECTOR>> VectorList;
    std::vector<std::complex<double>> EList;
    float minRealPart, maxImagPart;
    auto H = std::make_unique<sysMATRIX>();
    if (System == "CTIM") {
        *H = Hamiltonian(Bz, b);
    }
    else if (System == "CQHO") {
        *H = HamiltonianQHO(Bz, b);
    }

    // Solve for eigenvalues and eigenvectors
    auto solver = Eigen::ComplexEigenSolver<sysMATRIX>(*H);
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    std::complex<double> groundstateE = eigenvalues[0];
    std::complex<double> spectralE = eigenvalues[0];

    *Vector1 = eigenvectors.col(0);
    *GSVector1 = eigenvectors.col(0);
    minRealPart = groundstateE.real();
    maxImagPart = spectralE.imag();

    // Iterate through eigenvalues to find the one with the lowest real part
    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double realPart = eig.real();
        if (realPart < minRealPart) {
            minRealPart = realPart;
            groundstateE = eig;
            *GSVector1 = eigenvectors.col(i); // Get the corresponding eigenvector
            *GSVector1 /= GSVector1->norm();
        }
    }

    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double imagPart = eig.imag();
        if (imagPart > maxImagPart) {
            maxImagPart = imagPart;
            spectralE = eig;
            *Vector1 = eigenvectors.col(i); // Get the corresponding eigenvector 
        }
    }

    float xMagnetValue = XMagnet(*GSVector1);

    return xMagnetValue;
}

std::vector<float> runExactXmag(const std::vector<float> bList, const std::string Solver, const std::string System) {
	std::vector<float> XMag;
	for (const auto& b : bList) {
		std::cout << "=============" << std::endl;
		std::cout << "Variable: " << b << std::endl;
		float xMag = exactXmag(b, Solver, System);
		XMag.push_back(xMag);
		std::cout << "Xmag: " << xMag << std::endl;
	}
	return XMag;

}

float exactXSus(const float b, const std::string Solver, const std::string System) {
    float Bz = Constant; // This also acts as omega for the QHO
    auto Vector1 = std::make_unique<sysVECTOR>();
    auto GSVector1 = std::make_unique<sysVECTOR>();
    std::complex<double> E1;
    int iterations;
    std::vector<std::unique_ptr<sysVECTOR>> VectorList;
    std::vector<std::complex<double>> EList;
    float minRealPart, maxImagPart;
    auto H = std::make_unique<sysMATRIX>();
    if (System == "CTIM") {
        *H = Hamiltonian(Bz, b);
    }
    else if (System == "CQHO") {
        *H = HamiltonianQHO(Bz, b);
    }

    // Solve for eigenvalues and eigenvectors
    auto solver = Eigen::ComplexEigenSolver<sysMATRIX>(*H);
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();

    std::complex<double> groundstateE = eigenvalues[0];
    std::complex<double> spectralE = eigenvalues[0];

    *Vector1 = eigenvectors.col(0);
    *GSVector1 = eigenvectors.col(0);
    minRealPart = groundstateE.real();
    maxImagPart = spectralE.imag();

    // Iterate through eigenvalues to find the one with the lowest real part
    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double realPart = eig.real();
        if (realPart < minRealPart) {
            minRealPart = realPart;
            groundstateE = eig;
            *GSVector1 = eigenvectors.col(i); // Get the corresponding eigenvector
            *GSVector1 /= GSVector1->norm();
        }
    }

    for (int i = 0; i < eigenvalues.size(); ++i) {
        std::complex<double> eig = eigenvalues[i];
        double imagPart = eig.imag();
        if (imagPart > maxImagPart) {
            maxImagPart = imagPart;
            spectralE = eig;
            *Vector1 = eigenvectors.col(i); // Get the corresponding eigenvector 
        }
    }

    float xSusceptibilityValue = XSus(*GSVector1);

    return xSusceptibilityValue;
}

std::vector<float> runExactXSus(const std::vector<float> bList, const std::string Solver, const std::string System) {
    std::vector<float> XSus;
    for (const auto& b : bList) {
        std::cout << "=============" << std::endl;
        std::cout << "Variable: " << b << std::endl;
        float xSus = exactXSus(b, Solver, System);
        XSus.push_back(xSus);
        std::cout << "XSus: " << xSus << std::endl;
    }
    return XSus;
}

int exactOnly() {
    // Generate bList using linspace
    std::vector<float> bList = linspace(variableMin, variableMax, variablePoints);

    // Define Solver and System
    std::string Solver = "Exact";

    std::vector<float> XMag;

    // Run the exactXmag function over the bList
    if (Observ == "Xmag") {
        XMag = runExactXmag(bList, Solver, System);
    }
	else if (Observ == "Sus") {
	    XMag = runExactXSus(bList, Solver, System);
	}

    // Print bPoints
    std::cout << "bPointsExactCTIM = [" << std::endl;
    for (size_t i = 0; i < bList.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << bList[i];
        if (i != bList.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
       }
    }
    std::cout << std::endl << "]" << std::endl;

    // Print Xmag values
    std::cout << "TrueXMagCTIM" << n << " = [" << std::endl;
    for (size_t i = 0; i < XMag.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << XMag[i];
        if (i != XMag.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << "]" << std::endl;

    return 0;
}

int quantumAndExact() {
    std::vector<float> bList = linspace(variableMin, variableMax, variablePoints);
    std::vector<std::vector<std::complex<double>>> XmagPerBPoint(variablePoints, std::vector<std::complex<double>>(TOTALRUNS));
    std::vector<double> avgImagXmag(variablePoints, 0.0);
    std::vector<double> errorImagXmag(variablePoints, 0.0); // Vector to store the error (standard deviation)
    std::vector<double> avgIterations(variablePoints, 0.0); // Vector to store the average iterations
    std::vector<double> errorIterations(variablePoints, 0.0); // Vector to store the standard deviation of iterations
    std::tuple<std::vector<float>, std::vector<int>, std::vector<float>> XmagIterationsA;
    std::tuple<std::vector<float>, std::vector<int>> XmagIterations;
    std::vector<float> Xmag;
    std::vector<int> Iterations;
    std::vector<float> TrueXMag;
    for (int RUNS = 0; RUNS < TOTALRUNS; RUNS++) {
        if (Observ == "Xmag") {
            XmagIterationsA = runSolverXmag(bList, stepSize, gamma, firstStepIters, secondStepIters, "GD", System);
            Xmag = std::get<0>(XmagIterationsA);
            Iterations = std::get<1>(XmagIterationsA);
            TrueXMag = std::get<2>(XmagIterationsA);
        }
        else if (Observ == "Rmin") {
            XmagIterations = runSolverRmin(bList, stepSize, gamma, firstStepIters, secondStepIters, "GD", System);
            Xmag = std::get<0>(XmagIterations);
            Iterations = std::get<1>(XmagIterations);
        }
        else if (Observ == "Imax") {
            XmagIterations = runSolverImax(bList, stepSize, gamma, firstStepIters, secondStepIters, "GD", System);
            Xmag = std::get<0>(XmagIterations);
            Iterations = std::get<1>(XmagIterations);
        }
        else if (Observ == "Sus") {
            XmagIterationsA = runSolverXSus(bList, stepSize, gamma, firstStepIters, secondStepIters, "GD", System);
            Xmag = std::get<0>(XmagIterationsA);
            Iterations = std::get<1>(XmagIterationsA);
            TrueXMag = std::get<2>(XmagIterationsA);
        }
        for (size_t i = 0; i < bList.size(); ++i) {
            XmagPerBPoint[i][RUNS] = Xmag[i];
            avgIterations[i] += Iterations[i];
        }
    }

    // Compute the average and standard deviation of the absolute value of the imaginary part of Xmag for each bPoint
    for (size_t i = 0; i < bList.size(); ++i) {
        double sumImag = 0.0;
        for (int RUNS = 0; RUNS < TOTALRUNS; RUNS++) {
            sumImag += std::abs(XmagPerBPoint[i][RUNS]);
        }
        double avg = sumImag / static_cast<double>(TOTALRUNS);
        avgImagXmag[i] = avg;

        // Calculate variance for Xmag
        double varianceXmag = 0.0;
        for (int RUNS = 0; RUNS < TOTALRUNS; RUNS++) {
            varianceXmag += std::pow(std::abs(XmagPerBPoint[i][RUNS]) - avg, 2);
        }
        varianceXmag /= static_cast<double>(TOTALRUNS);
        errorImagXmag[i] = std::sqrt(varianceXmag); // Standard deviation as the error

        // Calculate average iterations
        avgIterations[i] /= static_cast<double>(TOTALRUNS);

        // Calculate variance for iterations
        double varianceIterations = 0.0;
        for (int RUNS = 0; RUNS < TOTALRUNS; RUNS++) {
            varianceIterations += std::pow(Iterations[RUNS] - avgIterations[i], 2);
        }
        varianceIterations /= static_cast<double>(TOTALRUNS);
        errorIterations[i] = std::sqrt(varianceIterations); // Standard deviation as the error
    }

    // Display the results in a table
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "bPoint\t\tAverage | " << Observ << " |\tError " << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
    for (size_t i = 0; i < bList.size(); ++i) {
        std::cout << bList[i] << "\t\t" << avgImagXmag[i] << "\t\t" << errorImagXmag[i] << std::endl;
    }
    // Print bPoints
    std::cout << "bPoints = [" << std::endl;
    for (size_t i = 0; i < bList.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << bList[i];
        if (i != bList.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << "]" << std::endl;

    // Print Sus
    std::cout << std::endl << "Sus = [" << std::endl;
    for (size_t i = 0; i < avgImagXmag.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << avgImagXmag[i];
        if (i != avgImagXmag.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << "]" << std::endl;

    // Print SusError
    std::cout << std::endl << "SusError = [" << std::endl;
    for (size_t i = 0; i < errorImagXmag.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << errorImagXmag[i];
        if (i != errorImagXmag.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << "]" << std::endl;

    // Print TrueXMag
    std::cout << std::endl << "TrueXMag = [" << std::endl;
    for (size_t i = 0; i < TrueXMag.size(); ++i) {
        std::cout << "    " << std::fixed << std::setprecision(4) << TrueXMag[i];
        if (i != TrueXMag.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl << "]" << std::endl;


    return 0;
}


int main() {
    if (exactOrQuantum == "Exact") {
        exactOnly();
    }
	else if (exactOrQuantum == "Quantum") {
		quantumAndExact();
	}  
	return 0;
}

