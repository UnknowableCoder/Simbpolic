#ifndef SIMBPOLIC_STORED
#define SIMBPOLIC_STORED

namespace Simbpolic
{
  template <indexer store_idx>
  struct Stored : public SymBase
  {
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return false;
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return true;
    }
            
    static constexpr indexer min_dimension = 0;
    static constexpr indexer max_dimension = 0;
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      return 0;
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return true;
    }
        
    friend std::ostream& operator << (std::ostream &s, const Stored& z)
    {
        s << "C_{" << store_idx << "}";
        return s;
    }

    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return (*this) * Monomial<1, dim>{};
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return Zero{};
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return (*this);
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() () const
    {
      return (*this);
    }
    
    template <class Store>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Store& store) const
    {
      if constexpr (is_store<Store>)
        {
          return Constant{store.template get<store_idx>()};
        }
      else
        {
          return (*this);
        }
    }
    
    template <class Store, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Store& store, const Args& ... args) const
    {
      if constexpr (is_store<Store>)
        {
          return Constant{store.template get<store_idx>()};
        }
      else
        {
          return (*this);
        }
    }
  };
  
  
}
#endif
