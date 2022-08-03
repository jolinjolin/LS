#include<iostream>
#include<Eigen/Sparse>
#include "linearSolver.hpp"

using namespace std;
using namespace Eigen;

int nx = 4, ny = 3, nz = 3;
int n = nx * ny * nz;
int m = ny * nz;

void initMats(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value){
	for(int i = 0; i < n; ++i){
		A.insert(i, i) = 1;
		if(i+1 >= 0 && i+1 < n){
			A.insert(i, i+1) = -1;
			A.insert(i+1, i) = -1;
			b[i] = 0.;
		}
	}

	int ix = nx-1;
	int idx = 0;
	int i = 0;
	for(int iy = 0; iy < ny; ++iy){
		for(int iz = 0; iz < nz; ++iz){
			idx = ix * ny * nz + iy * nz + iz;
			A.coeffRef(idx, idx) = 0;
			known_index[i] = idx;
			known_value[i] = 0;
			i++;
		}
	}

	ix = 0;
	for(int iy = 0; iy < ny; ++iy){
		for(int iz = 0; iz < nz; ++iz){
			idx = ix * ny * nz + iy * nz + iz;
			b[idx] = -0.01;
		}
	}

}

int main(){
	SparseMatrix<double> A(n,n);
	VectorXd b(n), x;	
	VectorXi known_index(m);
	VectorXd known_value(m);

	initMats(A, b, known_index, known_value);

	x = linear_solver<double>(A, b, known_index, known_value);


	for (int i = 0; i < x.rows(); ++i){
		cout << "x" << i << " = " << x(i) << endl;
	}
	return 0;
}