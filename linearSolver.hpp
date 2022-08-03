#include <Eigen/Eigen>
#include <Eigen/Sparse>

template<typename T>
Eigen::VectorXd linear_solver(const Eigen::SparseMatrix<T> &A, const Eigen::VectorXd &b, const Eigen::VectorXi &known_index, const Eigen::VectorXd &known_value)
{
    int n = b.rows();
    std::vector<bool> fixed(n, false);
    for (int i = 0; i < known_index.size(); ++i)
    {
        fixed[known_index[i]] = true;
    }
    std::vector<int> index_map(n);
    int num_fixed = 0;
    int num_unfixed = 0;
    for (int i = 0; i < n; ++i)
    {
        if (fixed[i])
        {
            index_map[i] = num_fixed;
            num_fixed += 1;
        }
        else
        {
            index_map[i] = num_unfixed;
            num_unfixed += 1;
        }
    }
    std::vector<Eigen::Triplet<double>> triplets1, triplets2;
    triplets1.reserve(A.nonZeros());
    triplets2.reserve(A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k)
    {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it)
        {
            int row = it.row();
            int col = it.col();
            if (!fixed[row])
            {
                if (!fixed[col])
                {
                    triplets1.push_back({index_map[row], index_map[col], it.value()});
                }
                else
                {
                    triplets2.push_back({index_map[row], index_map[col], it.value()});
                }
            }
        }
    }
    Eigen::SparseMatrix<T> A1(num_unfixed, num_unfixed);
    Eigen::SparseMatrix<T> A2(num_unfixed, num_fixed);
    A1.setFromTriplets(triplets1.begin(), triplets1.end());
    A2.setFromTriplets(triplets2.begin(), triplets2.end());

    Eigen::VectorXd b1 = Eigen::VectorXd::Zero(num_unfixed, 1);
    Eigen::VectorXd b2 = Eigen::VectorXd::Zero(num_fixed, 1);
    num_fixed = 0;
    num_unfixed = 0;
    for (int i = 0; i < n; ++i)
    {
        if (!fixed[i])
        {
            b1[num_unfixed] = b[i];
            num_unfixed += 1;
        }
        else
        {
            b2[num_fixed] = b[i];
            num_fixed += 1;
        }
    }
    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(A1);
    const Eigen::VectorXd &x2 = known_value;
    Eigen::VectorXd x1 = solver.solve(b1 - A2 * x2);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n, 1);
    num_fixed = 0;
    num_unfixed = 0;
    for (int i = 0; i < n; ++i)
    {
        if (!fixed[i])
        {
            x[i] = x1[num_unfixed];
            num_unfixed += 1;
        }
        else
        {
            x[i] = x2[num_fixed];
            num_fixed += 1;
        }
    }

    return x;
}
