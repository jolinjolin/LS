#include<iostream>
#include<Eigen/Sparse>
#include "linearSolver.hpp"

using namespace std;
using namespace Eigen;

int nx = 4, ny = 3, nz = 3;
int n = nx * ny * nz;
int m = ny * nz;

void initMats(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value){
	vector<vector<int>> direct = {
			{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, 
			{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 
			{1, 1, 0}, {0, 1, 1}, {1, 0, 1}, 
			{-1, 1, 0}, {-1, -1, 0}, {1, 1, 0}, 
			{1, -1, 0}, {-1, 0, 1}, {-1, 0, -1}, 
			{1, 0, 1}, {1, 0, -1}, {0, -1, 1},
			 {0, -1, -1}, {0, 1, 1}, {0, 1, -1}
	};
	//fill A non-zeros
	int idx = 0;
	for(int ix = 0; ix < nx; ++ix){
		for(int iy = 0; iy < ny; ++iy){
			for(int iz = 0; iz < ny; ++iz){
				idx = ix * ny * nz + iy * nz + iz;
				A.coeffRef(idx, idx) += 1;
				for(int j = 0; j < direct.size(); ++j){
					int jx = direct[j][0] + ix, jy = direct[j][1] + iy, jz = direct[j][1] + iz;
					if(jx >= 0 && jx < nx && jy >= 0 && jy < ny && jz >= 0 && jz < nz){
						int jdx = jx * ny * nz + jy * nz + jz;
						A.coeffRef(jdx, jdx) += 1;
						A.coeffRef(idx, jdx) += -1;
					}

				}
			}
		}
	}

	//fill A zeros(BC)
	int i = 0;
	for(int ix = nx-1; ix < nx; ++ix){
		for(int iy = 0; iy < ny; ++iy){
			for(int iz = 0; iz < nz; ++iz){
				idx = ix * ny * nz + iy * nz + iz;
				A.coeffRef(idx, idx) = 0;
				known_index[i] = idx;
				known_value[i] = 0;
				i++;
			}
		}
	}

	//fill b zeros and nonzeros
	for(int ix = 0; ix < 1; ++ix){
		for(int iy = 0; iy < ny; ++iy){
			for(int iz = 0; iz < nz; ++iz){
				idx = ix * ny * nz + iy * nz + iz;
				if(ix == 0){
					b[idx] = -0.01;
				}
				else{
					b[idx] = 0;
				}
			}
		}
	}

}

int main(){
	SparseMatrix<double> A(n,n);
	VectorXd b(n), x;	
	VectorXi known_index(m);
	VectorXd known_value(m);

	initMats(A, b, known_index, known_value);
	// std::cout << MatrixXd(A) << std::endl;

	x = linear_solver<double>(A, b, known_index, known_value);


	for (int i = 0; i < x.rows(); ++i){
		cout << "x" << i << " = " << x(i) << endl;
	}

	return 0;
}