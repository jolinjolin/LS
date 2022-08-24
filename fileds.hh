#if !defined(FIELDS)
#define FIELDS

namespace plb
{
  template <typename T, template <typename U1> class ADESCRIPTOR>
  class calFroce : public BoxProcessingFunctional3D
  {
  public:
    calFroce(T dx_, T k_n _) : dx(dx_), T k_n(k_n_)
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      ScalarField3D<T, 3> &displaceOld = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[0]);
      ScalarField3D<T, 3> &displaceNew = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[1]);
      ScalarField3D<T, 3> &force = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[2]);

      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            Array<T, 3> u_i(0., 0., 0.), u_j(0., 0., 0.), u_ij(0., 0., 0.);
            u_i = Array<T, 3>(displaceOld.get(x0, y0, z0)[0], displaceOld.get(x0, y0, z0)[1], displaceOld.get(x0, y0, z0)[2]);
            for (plint i = 1; i < ADESCRIPTOR<T>::q; ++i)
            {
              plint x1 = x0 + ADESCRIPTOR<T>::c[i][0], y1 = y0 + ADESCRIPTOR<T>::c[i][1], z1 = z0 + ADESCRIPTOR<T>::c[i][2];
              u_i = Array<T, 3>(displaceOld.get(x1, y1, z1)[0], displaceOld.get(x1, y1, z1)[1], displaceOld.get(x1, y1, z1)[2]);
              u_ij = Array<T, 3>(u_j[0] - u_i[0], u_j[1] - u_i[1], u_j[2] - u_i[2]);
              T u_ij_sq = u_ij[0] * u_ij[0] + u_ij[1] * u_ij[1] + u_ij[2] * u_ij[2];
              if (u_ij_sq < 1.732 * 1.732 * dx * dx)
              {
                T norm = T(ADESCRIPTOR<T>::c[i][0] * ADESCRIPTOR<T>::c[i][0] + ADESCRIPTOR<T>::c[i][1] * ADESCRIPTOR<T>::c[i][1] + ADESCRIPTOR<T>::c[i][2] * ADESCRIPTOR<T>::c[i][2]);
                norm = sqrt(norm);
                Array<T, 3> norm_unit((T)ADESCRIPTOR<T>::c[i][0] / norm, (T)ADESCRIPTOR<T>::c[i][1] / norm, (T)ADESCRIPTOR<T>::c[i][2] / norm);
                T dot_product = u_ij[0] * norm_unit[0] + u_ij[1] * norm_unit[1] + u_ij[2] * norm_unit[2];
                Array<T, 3> u_ij_n(dot_product * norm_unit[0] + dot_product * norm_unit[1] + dot_product * norm_unit[2]);
                force.get(x0, y0, z0)[0] += k_n * u_ij_n[0];
                force.get(x0, y0, z0)[1] += k_n * u_ij_n[1];
                force.get(x0, y0, z0)[2] += k_n * u_ij_n[2];
              }
            }
          }
    }
    //...........................................................................................................................
    virtual calFroce<T, ADESCRIPTOR> *clone() const
    {
      return new calFroce<T, ADESCRIPTOR>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[0] = modif::nothing;
      modified[1] = modif::staticVariables;
      modified[2] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
    T dx, k_n;
  };
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, template <typename U1> class ADESCRIPTOR>
  class verletUpdate : public BoxProcessingFunctional3D
  {
  public:
    verletUpdate(T dx_, T k_n _) : dx(dx_), T k_n(k_n_)
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      ScalarField3D<T, 3> &displaceOld = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[0]);
      ScalarField3D<T, 3> &displaceNew = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[1]);
      ScalarField3D<T, 3> &force = *dynamic_cast<ScalarField3D<T, 3> *>(blocks[2]);

      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            T damping_factor = 0.8;
            T sign;
            Array<T, 3> acc = force[idx] / mass
            for(plint d = 0; d < 3; ++d){
						  sign = (displaceNew[d]-displaceOld[d]) >= 0 ? 1.0 : -1.0;
              displaceOld.get(x0, y0, z0)[d] = (displaceNew.get(x0, y0, z0)[d] + acc[d] * 0.5);
              force[idx][d] = force_change[d];
              if ((i == NX - 1) && d == 0)
              {
                force[idx][d] += init_force;
              }
              damping_acc[d] = (force[idx][d] - damping_factor * fabs(force[idx][d]) * sign) * rmass;
              velocity[idx][d] += damping_acc[d] * 0.5;
            }
            
          }
    }
    //...........................................................................................................................
    virtual verletUpdate<T, ADESCRIPTOR> *clone() const
    {
      return new verletUpdate<T, ADESCRIPTOR>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[0] = modif::staticVariables;
      modified[1] = modif::staticVariables;
      modified[2] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
    T dx, k_n;
  };
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace plb

#endif // FIELDS
