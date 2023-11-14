#include <iostream>
#include <Eigen/Sparse>
#include "linearSolver.hpp"

#include <vector>
#include "palabos3D.h"
#include "palabos3D.hh"

#include "lsFields.hh"

using namespace plb;
using namespace std;
using namespace Eigen;

typedef double T;
#define D 3
#define Q 19

#define ADESCRIPTOR descriptors::AdvectionDiffusionD3Q19Descriptor
#define ADYNAMICS AdvectionDiffusionBGKdynamics

MultiTensorField3D<T, D> *displace, *forceOld, *forceNew, *velocity;
std::vector<MultiBlock3D *> dataField;

int nx, ny, nz, n, m;
T dx, dt;
T k_n, mass;
Box3D domain;
plb::Array<T, D> init_force;

T damping_coeff;

vector<vector<int>> direct = {
	{-1, 0, 0},
	{0, -1, 0},
	{0, 0, -1},
	{1, 0, 0},
	{0, 1, 0},
	{0, 0, 1},
	{1, 1, 0},
	{0, 1, 1},
	{1, 0, 1},
	{-1, -1, 0},
	{0, -1, -1},
	{-1, 0, -1},
	{-1, 1, 0},
	{-1, 0, 1},
	{0, -1, 1},
	{1, -1, 0},
	{1, 0, -1},
	{0, 1, -1},
};

void init_param()
{
	nx = 4;
	ny = 4;
	nz = 4;
	n = nx * ny * nz;
	m = ny * nz;

	dx = 0.01;
	dt = 0.001;
	k_n = 1.0;
	mass = 1.0;

	damping_coeff = 0.003;

	init_force = plb::Array<T, D>(1e-3, 0., 0.);

	domain = Box3D(0, nx - 1, 0, ny - 1, 0, nz - 1);
}

void create_field()
{
	displace = new MultiTensorField3D<T, D>(nx, ny, nz);
	forceOld = new MultiTensorField3D<T, D>(nx, ny, nz);
	forceNew = new MultiTensorField3D<T, D>(nx, ny, nz);
	velocity = new MultiTensorField3D<T, D>(nx, ny, nz);

	displace->periodicity().toggle(0, true);
	displace->periodicity().toggle(1, true);
	displace->periodicity().toggle(2, true);
	forceOld->periodicity().toggle(0, true);
	forceOld->periodicity().toggle(1, true);
	forceOld->periodicity().toggle(2, true);
	forceNew->periodicity().toggle(0, true);
	forceNew->periodicity().toggle(1, true);
	forceNew->periodicity().toggle(2, true);
	velocity->periodicity().toggle(0, true);
	velocity->periodicity().toggle(1, true);
	velocity->periodicity().toggle(2, true);
}
void init_arg()
{
	dataField.push_back(displace);
	dataField.push_back(forceOld);
	dataField.push_back(forceNew);
	dataField.push_back(velocity);
}
void init_field()
{
	applyProcessingFunctional(new initializeFieldsLS<T>(), domain, dataField);
}
void ls_motion(plint iT)
{
	applyProcessingFunctional(new updateFieldsLS<T>(init_force), domain, dataField);
	applyProcessingFunctional(new calFroceLS<T, ADESCRIPTOR>(dx, dt, k_n, iT), domain, dataField);
	applyProcessingFunctional(new verletUpdateLS<T>(dx, dt, mass, iT), domain, dataField);
}
void output_data_field(plint iT) ß
{
	if (iT % 1000 == 0)
	{
		VtkImageOutput3D<T> VtkOut00(createFileName("displacement", iT, 7), 1.);
		VtkOut00.writeData<D, T>(*displace, "displacement", 1.);
	}
}
void copy_to_field(VectorXd x)
{
	int idx = 0;
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				if (isnan(x[idx]))
				{
					displace->get(ix, iy, iz)[0] = 0.;
				}
				else
				{
					displace->get(ix, iy, iz)[0] = x[idx];
				}
			}
		}
	}
}
void clean_up()
{
	delete displace;
	delete forceOld;
	delete forceNew;
	delete velocity;
}

// void initMats0(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value)
// {
// 	// fill A non-zeros
// 	int idx = 0;
// 	for (int ix = 0; ix < nx; ++ix)
// 	{
// 		for (int iy = 0; iy < ny; ++iy)
// 		{
// 			for (int iz = 0; iz < ny; ++iz)
// 			{
// 				idx = ix * ny * nz + iy * nz + iz;
// 				A.coeffRef(idx, idx) += k_n;
// 				for (int j = 0; j < direct.size(); ++j)
// 				{
// 					int jx = direct[j][0] + ix;
// 					int jy = 0, jz = 0;
// 					if (jx >= 0 && jx < nx)
// 					{
// 						jy = (direct[j][1] + iy + ny) % ny, jz = (direct[j][2] + iz + nz) % nz;
// 						int jdx = jx * ny * nz + jy * nz + jz;
// 						A.coeffRef(jdx, jdx) += k_n;
// 						A.coeffRef(idx, jdx) += -k_n;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	// fill A zeros(BC)
// 	int i = 0;
// 	for (int ix = nx - 1; ix < nx; ++ix)
// 	{
// 		for (int iy = 0; iy < ny; ++iy)
// 		{
// 			for (int iz = 0; iz < nz; ++iz)
// 			{
// 				idx = ix * ny * nz + iy * nz + iz;
// 				A.coeffRef(idx, idx) = 0;
// 				known_index[i] = idx;
// 				known_value[i] = 0;
// 				i++;
// 			}
// 		}
// 	}

// 	// fill b zeros and nonzeros
// 	for (int ix = 0; ix < nx; ++ix)
// 	{
// 		for (int iy = 0; iy < ny; ++iy)
// 		{
// 			for (int iz = 0; iz < nz; ++iz)
// 			{
// 				idx = ix * ny * nz + iy * nz + iz;
// 				if (ix == nx-1)
// 				{
// 					b[idx] = 0.;
// 				}
// 				else {
// 					b[idx] = damping_coeff * displace->get(ix, iy, iz)[0];
// 					if(ix == 0) {
// 						b[idx] += init_force[0];
// 					}
// 				}
// 			}
// 		}
// 	}
// }

// void updateMats0(SparseMatrix<double> &A, VectorXd &b, VectorXi &known_index, VectorXd &known_value)
// {
// 	int idx = 0;
// 	for (int ix = 0; ix < nx-1; ++ix)
// 	{
// 		for (int iy = 0; iy < ny; ++iy)
// 		{
// 			for (int iz = 0; iz < nz; ++iz)
// 			{
// 				idx = ix * ny * nz + iy * nz + iz;
// 				b[idx] = damping_coeff * displace->get(ix, iy, iz)[0];
// 				// b[idx] = 0.;
// 				if (ix == 0)
// 				{
// 					b[idx] += init_force[0];
// 				}
// 			}
// 		}
// 	}
// }

void initMats(SparseMatrix<double> &K, SparseMatrix<double> &C, SparseMatrix<double> &M, VectorXd &x, VectorXd &displace_minus1, VectorXd &displace_minus2, VectorXd &displace_minus3, VectorXd &displace_minus4)
{
	// fill K, C symmetric matrices
	int idx = 0;
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < ny; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				K.coeffRef(idx, idx) += k_n;
				C.coeffRef(idx, idx) += damping_coeff;
				M.coeffRef(idx, idx) += mass;
				x[idx] = 0.;
				displace_minus1[idx] = 0.;
				displace_minus2[idx] = 0.;
				displace_minus3[idx] = 0.;
				displace_minus4[idx] = 0.;
				for (int j = 0; j < direct.size(); ++j)
				{
					int jx = direct[j][0] + ix;
					int jy = 0, jz = 0;
					if (jx >= 0 && jx < nx)
					{
						jy = (direct[j][1] + iy + ny) % ny, jz = (direct[j][2] + iz + nz) % nz;
						int jdx = jx * ny * nz + jy * nz + jz;
						K.coeffRef(jdx, jdx) += k_n;
						K.coeffRef(idx, jdx) += -k_n;
						C.coeffRef(jdx, jdx) += damping_coeff;
						C.coeffRef(idx, jdx) += -damping_coeff;
					}
				}
			}
		}
	}
}

void updateDisplacement(VectorXd &displace_minus1, VectorXd &displace_minus2, VectorXd &displace_minus3, VectorXd &displace_minus4)
{
	int idx = 0;
	for (int ix = 0; ix < nx - 1; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < ny; ++iz)
			{
				idx = ix * ny * nz + iy * nz + iz;
				displace_minus4[idx] = displace_minus3[idx];
				displace_minus3[idx] = displace_minus2[idx];
				displace_minus2[idx] = displace_minus1[idx];
				displace_minus1[idx] = displace->get(ix, iy, iz)[0];
			}
		}
	}
}

void calDisplacement(SparseMatrix<double> &K, SparseMatrix<double> &C, SparseMatrix<double> &M, VectorXd &x, VectorXd &displace_minus1, VectorXd &displace_minus2, VectorXd &displace_minus3, VectorXd &displace_minus4)
{
	auto sm1 = -(4 * K - 2 * M);
	auto sm2 = -2 * C;
	x = sm1 * displace_minus2 + sm2 * displace_minus1 - sm2 * displace_minus3 - M * displace_minus4;
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				int idx = ix * ny * nz + iy * nz + iz;
				if (ix == 0)
				{
					x[idx] += init_force[0] / mass;
				}
				if (ix == nx - 1)
				{
					x[idx] = 0.;
				}
			}
		}
	}
}
void outputMats(VectorXd &x)
{
	for (int ix = 0; ix < nx; ++ix)
	{
		for (int iy = 0; iy < ny; ++iy)
		{
			for (int iz = 0; iz < nz; ++iz)
			{
				int idx = ix * ny * nz + iy * nz + iz;
				pcout << ix << " " << iy << " " << iz << " " << x[idx] << endl;
			}
		}
	}
}

int main()
{
	global::directories().setOutputDir("./tmp/");
	init_param();
	create_field();
	init_arg();
	init_field();

	SparseMatrix<double> K(n, n), C(n, n), M(n, n);
	VectorXd x(n), displace_minus1(n), displace_minus2(n), displace_minus3(n), displace_minus4(n);
	initMats(K, C, M, x, displace_minus1, displace_minus2, displace_minus3, displace_minus4);

	for (int iT = 0; iT <= 20; ++iT)
	{
		// x = linear_solver<double>(A, b, known_index, known_value);
		calDisplacement(K, C, M, x, displace_minus1, displace_minus2, displace_minus3, displace_minus4);
		copy_to_field(x);
		ls_motion(iT);
		updateDisplacement(displace_minus1, displace_minus2, displace_minus3, displace_minus4);
		// output_data_field(iT);
		if (iT % 10 == 0)
		{
			pcout << iT << endl;
			outputMats(x);
		}
	}

	clean_up();
	return 0;
}