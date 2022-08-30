#if !defined(LSFIELDS)
#define LSFIELDS

using namespace std;

namespace plb
{
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  class initializeFieldsLS : public BoxProcessingFunctional3D
  {
  public:
    initializeFieldsLS()
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      TensorField3D<T, 3> &displace = *dynamic_cast<TensorField3D<T, 3> *>(blocks[0]);
      TensorField3D<T, 3> &forceOld = *dynamic_cast<TensorField3D<T, 3> *>(blocks[1]);
      TensorField3D<T, 3> &forceNew = *dynamic_cast<TensorField3D<T, 3> *>(blocks[2]);
      TensorField3D<T, 3> &velocity = *dynamic_cast<TensorField3D<T, 3> *>(blocks[3]);

      Dot3D offset = displace.getLocation();
      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            displace.get(x0, y0, z0) = Array<T,3>(0., 0., 0.);
            forceOld.get(x0, y0, z0) = Array<T,3>(0., 0., 0.);
            forceNew.get(x0, y0, z0) = Array<T,3>(0., 0., 0.);
            velocity.get(x0, y0, z0) = Array<T,3>(0., 0., 0.);
          }
    }
    //...........................................................................................................................
    virtual initializeFieldsLS<T> *clone() const
    {
      return new initializeFieldsLS<T>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[0] = modif::staticVariables;
      modified[1] = modif::staticVariables;
      modified[2] = modif::staticVariables;
      modified[3] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
  };
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  class updateFieldsLS : public BoxProcessingFunctional3D
  {
  public:
    updateFieldsLS(Array<T, 3> init_force_) : init_force(init_force_)
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      TensorField3D<T, 3> &forceNew = *dynamic_cast<TensorField3D<T, 3> *>(blocks[2]);

      Dot3D offset = forceNew.getLocation();
      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            if (x0 + offset.x == 0)
            {
              forceNew.get(x0, y0, z0) += init_force;
            }
            else
            {
              forceNew.get(x0, y0, z0) = Array<T, 3>(0., 0., 0.);
            }
          }
    }
    //...........................................................................................................................
    virtual updateFieldsLS<T> *clone() const
    {
      return new updateFieldsLS<T>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[2] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
    Array<T, 3> init_force;
  };
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T, template <typename U1> class ADESCRIPTOR>
  class calFroceLS : public BoxProcessingFunctional3D
  {
  public:
    calFroceLS(T dx_,T dt_, T k_n_) : dx(dx_), dt(dt_), k_n(k_n_)
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      TensorField3D<T, 3> &displace = *dynamic_cast<TensorField3D<T, 3> *>(blocks[0]);
      TensorField3D<T, 3> &forceNew = *dynamic_cast<TensorField3D<T, 3> *>(blocks[2]);

      Dot3D offset = displace.getLocation();
      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            Array<T, 3> u_i(0., 0., 0.), u_j(0., 0., 0.), u_ij(0., 0., 0.);
            u_i = displace.get(x0, y0, z0);
            for (plint i = 1; i < ADESCRIPTOR<T>::q; ++i)
            {
              plint x1 = x0 + ADESCRIPTOR<T>::c[i][0], y1 = y0 + ADESCRIPTOR<T>::c[i][1], z1 = z0 + ADESCRIPTOR<T>::c[i][2];
              u_j = displace.get(x1, y1, z1);
              u_ij = Array<T, 3>(u_j[0] - u_i[0], u_j[1] - u_i[1], u_j[2] - u_i[2]);
              T u_ij_sq = u_ij[0] * u_ij[0] + u_ij[1] * u_ij[1] + u_ij[2] * u_ij[2];
              if (u_ij_sq < 1.732 * 1.732 * dx * dx)
              {
                T norm = T(ADESCRIPTOR<T>::c[i][0] * ADESCRIPTOR<T>::c[i][0] + ADESCRIPTOR<T>::c[i][1] * ADESCRIPTOR<T>::c[i][1] + ADESCRIPTOR<T>::c[i][2] * ADESCRIPTOR<T>::c[i][2]);
                norm = sqrt(norm);
                Array<T, 3> norm_unit((T)ADESCRIPTOR<T>::c[i][0] / norm, (T)ADESCRIPTOR<T>::c[i][1] / norm, (T)ADESCRIPTOR<T>::c[i][2] / norm);
                T dot_product = u_ij[0] * norm_unit[0] + u_ij[1] * norm_unit[1] + u_ij[2] * norm_unit[2];
                Array<T, 3> u_ij_n = dot_product * norm_unit;
                forceNew.get(x0, y0, z0)[0] += k_n * u_ij_n[0];
                forceNew.get(x0, y0, z0)[1] += k_n * u_ij_n[1];
                forceNew.get(x0, y0, z0)[2] += k_n * u_ij_n[2];
              }
            }
          }
    }
    //...........................................................................................................................
    virtual calFroceLS<T, ADESCRIPTOR> *clone() const
    {
      return new calFroceLS<T, ADESCRIPTOR>(*this);
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
    T dx, dt, k_n;
  };
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  class verletUpdateLS : public BoxProcessingFunctional3D
  {
  public:
    verletUpdateLS(T dx_, T dt_, T mass_) : dx(dx_), dt(dt_), mass(mass_)
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {
      TensorField3D<T, 3> &displace = *dynamic_cast<TensorField3D<T, 3> *>(blocks[0]);
      TensorField3D<T, 3> &forceOld = *dynamic_cast<TensorField3D<T, 3> *>(blocks[1]);
      TensorField3D<T, 3> &forceNew = *dynamic_cast<TensorField3D<T, 3> *>(blocks[2]);
      TensorField3D<T, 3> &velocity = *dynamic_cast<TensorField3D<T, 3> *>(blocks[3]);

      T damping_factor = 0.8;
      T sign;
      T rmass = 1.0 / mass;
      Dot3D offset = displace.getLocation();
      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            if(x0 + offset.x == 0){
              // pcout << forceNew.get(x0, y0, z0)[0] << " " << forceNew.get(x0, y0, z0)[1] << " " << forceNew.get(x0, y0, z0)[2] << endl ;
            }
            Array<T, 3> acc, acc_new, vel;
            acc = forceOld.get(x0, y0, z0) * rmass;
            vel = velocity.get(x0, y0, z0);
            for (plint d = 0; d < 3; ++d)
            {
              displace.get(x0, y0, z0)[d] += vel[d] * dt + acc[d] * 0.5 * dt * dt;
              // sign = (vel[d]) >= 0 ? 1.0 : -1.0;
              // acc_new[d] = (forceNew.get(x0, y0, z0)[d] - damping_factor * fabs(forceNew.get(x0, y0, z0)[d]) * sign) * rmass;
              acc_new[d] = forceNew.get(x0, y0, z0)[d] * rmass;
              vel[d] += 0.5 * (acc[d] + acc_new[d]) * 0.5 * dt;
            }
            velocity.get(x0, y0, z0) = vel;
            forceOld.get(x0, y0, z0) = forceNew.get(x0, y0, z0);
          }
    }
    //...........................................................................................................................
    virtual verletUpdateLS<T> *clone() const
    {
      return new verletUpdateLS<T>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[0] = modif::staticVariables;
      modified[1] = modif::staticVariables;
      modified[2] = modif::staticVariables;
      modified[3] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
    T dx, dt, mass;
  };
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace plb

#endif // FIELDS
