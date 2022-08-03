#include<iostream>
#include<Eigen/Sparse>
#include "linearSolver.hpp"

using namespace std;
using namespace Eigen;

void initMats(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value){
	A.insert(0, 0) = 1;
	A.insert(0, 1) = 2;
	A.insert(1, 0) = 2;
    A.insert(1, 1) = 2;
	A.insert(1, 3) = 1;
	A.insert(3, 1) = 3;
	A.insert(3, 3) = 4;
	
	A.insert(2, 2) = 0;
	A.insert(4, 4) = 0;

	known_index[0] = 2;
	known_index[1] = 4;
	known_value[0] = 0;
	known_value[1] = 0;

	b << 1, 4, 0, 2, 0;
}

int main(){
    int n = 5, m = 2;
	SparseMatrix<double> A(n,n);
	VectorXd b(n), x;	
	VectorXi known_index(m);
	VectorXd known_value(m);

	initMats(A, b, known_index, known_value);

	// SparseLU<SparseMatrix<double>> solver;
	// solver.compute(A);
	// x = solver.solve(b);

	x = linear_solver<double>(A, b, known_index, known_value);


	for (int i = 0; i < x.rows(); i++){
		cout << "x" << i << " = " << x(i) << endl;
	}
	return 0;
}