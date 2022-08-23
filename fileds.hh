#if !defined(FIELDS)
#define FIELDS

namespace plb
{
 template <typename T>
  class verletUpdate : public BoxProcessingFunctional3D
  {
  public:
    verletUpdate()
    {
    }
    virtual void processGenericBlocks(Box3D domain, vector<AtomicBlock3D *> blocks)
    {      
      ScalarField3D<bool> &displaceOld = *dynamic_cast<ScalarField3D<bool> *>(blocks[0]);
      ScalarField3D<bool> &displaceNew = *dynamic_cast<ScalarField3D<bool> *>(blocks[1]);

      for (plint x0 = domain.x0; x0 <= domain.x1; ++x0)
        for (plint y0 = domain.y0; y0 <= domain.y1; ++y0)
          for (plint z0 = domain.z0; z0 <= domain.z1; ++z0)
          {
            
          }
    }
    //...........................................................................................................................
    virtual updateRho<T, ADESCRIPTOR> *clone() const
    {
      return new updateRho<T, ADESCRIPTOR>(*this);
    }

    virtual void getTypeOfModification(std::vector<modif::ModifT> &modified) const
    {
      modified[0] = modif::staticVariables;
      modified[1] = modif::staticVariables;
    }

    virtual BlockDomain::DomainT appliesTo() const
    {
      return BlockDomain::bulk;
    }

  private:
    T factor;
  };
} // namespace plb



#endif // FIELDS
