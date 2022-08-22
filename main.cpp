#include <iostream>
#include <Eigen/Sparse>
#include "linearSolver.hpp"

#include <vector>
#include "palabos3D.h"
#include "palabos3D.hh"

using namespace plb;
using namespace std;
using namespace Eigen;

typedef double T;
#define D 3
#define Q 18

#define ADESCRIPTOR descriptors::AdvectionDiffusionD3Q19Descriptor
#define ADYNAMICS AdvectionDiffusionBGKdynamics

MultiTensorField3D<T, D> *displacement;
std::vector<MultiBlock3D *> dataField;

int nx, ny, nz, n, m;
Box3D domain;

void init_param()
{
    nx = 4;
    ny = 3;
    nz = 3;
    n = nx * ny * nz;
	m = ny * nz;

    domain = Box3D(0,nx-1, 0, ny-1, 0, nz-1);

}

void create_field()
{
	displacement = new MultiTensorField3D<T, D>(nx, ny, nz);

	displacement->periodicity().toggle(0, false);
    displacement->periodicity().toggle(1, false);
    displacement->periodicity().toggle(2, false);
}
void init_arg()
{

}
void output_data_field(plint iT)
{
    VtkImageOutput3D<T> VtkOut00(createFileName("displacement", iT, 7), 1.);
    VtkOut00.writeData<D, T>(*displacement, "displacement", 1.);
}
void copy_to_field(VectorXd x)
{
	int idx = 0;
	for(int ix = 0; ix < nx; ++ix){
		for(int iy = 0; iy < ny; ++iy){
			for(int iz = 0; iz < nz; ++iz){
				idx = ix * ny * nz + iy * nz + iz;
				if(isnan(x[idx])){
					displacement->get(ix, iy, iz)[0] = 0.;
				}
				else{
					displacement->get(ix, iy, iz)[0] = x[idx];
				}
			}
		}
	}
}
void clean_up()
{
    delete displacement;
}

void initMats(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value)
{
	vector<vector<int>> direct = {
		{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, 
		{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 
		{1, 1, 0}, {0, 1, 1}, {1, 0, 1},
		{-1, -1, 0}, {0, -1, -1}, {-1, 0, -1}, 
		{-1, 1, 0}, {-1, 0, 1}, {0, -1, 1},
		{1, -1, 0}, {1, 0, -1}, {0, 1, -1}, 
	};
	// fill A non-zeros
	int idx = 0;
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < ny; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				A.coeffRef(idx, idx) += 1;
				for (int j = 0; j < direct.size(); ++j)
				{
					int jx = direct[j][0] + ix, jy = direct[j][1] + iy, jz = direct[j][2] + iz;
					if (jx >= 0 && jx < nx && jy >= 0 && jy < ny && jz >= 0 && jz < nz)
					{
						int jdx = jx * ny * nz + jy * nz + jz;
						A.coeffRef(jdx, jdx) += 1;
						A.coeffRef(idx, jdx) += -1;
					}
				}
			}
		}
	}

	// fill A zeros(BC)
	int i = 0;
	for (int ix = nx - 1; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				A.coeffRef(idx, idx) = 0;
				known_index[i] = idx;
				known_value[i] = 0;
				i++;
			}
		}
	}

	// fill b zeros and nonzeros
	for (int ix = 0; ix < 1; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				if (ix == 0)
				{
					b[idx] = 0.01;
				}
				else
				{
					b[idx] = 0;
				}
			}
		}
	}
}

int main()
{
	global::directories().setOutputDir("./tmp/");
	init_param();

	SparseMatrix<double> A(n, n);
	VectorXd b(n), x;
	VectorXi known_index(m);
	VectorXd known_value(m);

	initMats(A, b, known_index, known_value);
	// std::cout << MatrixXd(A) << std::endl;

	x = linear_solver<double>(A, b, known_index, known_value);

	create_field();
	copy_to_field(x);

	// for (int i = 0; i < x.rows(); ++i)
	// {
	// 	cout << "x" << i << " = " << x(i) << endl;
	// }

	// for (int ix = 0; ix < nx; ++ix)
	// {
	// 	for (int iy = 0; iy < ny; ++iy)
	// 	{
	// 		for (int iz = 0; iz < nz; ++iz)
	// 		{
	// 			int idx = ix * ny * nz + iy * nz + iz;
	// 			pcout << ix << " " << iy << " " << iz << " " << x[idx] << endl;

	// 		}
	// 	}
	// }

	output_data_field(0);
	clean_up();
	return 0;
}