#ifndef SIMBPOLIC_MONOMIAL
#define SIMBPOLIC_MONOMIAL

namespace Simbpolic
{

  template <indexer order, indexer dim> struct Monomial : public SymBase
  {
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return dim == dimension;
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return order == 0;
    }
        
    static constexpr indexer min_dimension = dim;
    static constexpr indexer max_dimension = dim;
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      if constexpr (dimension == dim)
        {
          if constexpr (order < -1)
            {
              return 1;
            }
          else if constexpr (order == -1)
            {
              return std::numeric_limits<indexer>::max()/2;
              //Just to be a high value,
              //currently we do not support logarithms (yet)
            }
          else
            {
              return 1;
            }
        }
      else
        {
          return 1;
        }
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return true;
    }
    
    friend std::ostream& operator << (std::ostream &s, const Monomial& z)
    {
        s << " x_{" << dim << "} ^ " << order;
        return s;
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr Monomial()
    {
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      if constexpr(dimension == dim)
        {
          static_assert(order != -1, "Logarithms aren't currently supported!");
          /*if constexpr (order == -1)
            {
              return Functions::log(Functions::abs(monomial<1>{1}));
            }          
          else
            {*/
              return Rational<1, order+1>{} * Monomial<order+1, dim>{};
            /*}*/
        }
      else
        {
          return Monomial<1, dimension>{} * (*this);
        }
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      if constexpr (dimension == dim && order != 0)
        {
          if constexpr (order == 1)
            {
              return One{};
            }
          else
            {
              return Rational<order, 1>{} * Monomial<order-1, dim>{};
            }
        }
      else
        {
          return Zero{};
        }
    }
    
    template <indexer dummy = order, typename std::enable_if_t<dummy == 0>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator One () const
    {
      return One{};
    }
    
    template <indexer dummy = order, typename std::enable_if_t<dummy == 1>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Var<dim> () const
    {
      return Var<dim>{};
    }
    
    private:
    
    template <class F, typename std::enable_if_t<is_symbolic<F> || is_numeric<F>>* = nullptr >
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate (const F& f) const
    {
      if constexpr (!is_symbolic<F>)
        {
          return Constant{fastpow(Type(f), order)};
        }
      else
        {
          return ( f ^ Intg<order>{} ); 
        }
    }
    
    template <indexer idx>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto eat_args() const
    {
      return (*this);
    }
    
    template <indexer idx, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto eat_args(const Arg& arg) const
    {
     if constexpr (idx == dim)
        {
          return evaluate(arg);
        }
      else
        {
          return (*this);
        }
    }
    
    template <indexer idx, class First, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto eat_args (const First& f, const Args& ... args) const
    {
      if constexpr (idx > dim)
        {
          return (*this);
        }
      else if constexpr (idx == dim)
        {
          return evaluate(f);
        }
      else
        {
          return eat_args<idx + 1>(args...);
        }
    }
    
    public:
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() () const
    {
      return (*this);
    }
    
    template <class Arg, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Arg& first, const Args& ... args) const
    {
      if constexpr (is_store<Arg>)
        {
          return eat_args<1>(args...);
        }
      else
        {
          return eat_args<1>(first, args...);
        }
    }
    
    template <indexer dimension, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim (const Arg& val) const
    {
      if constexpr (dimension == dim)
        {
          return evaluate(val);
        }
      else
        {
          return (*this);
        }
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      if constexpr (from == dim)
        {
          return Monomial<order, to>{};
        }
      else
        {
          return (*this);
        }
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      if constexpr (dimension == dim)
        {
          return this->evaluate(Monomial<order, 1>{} + off);
        }
      else
        {
          return (*this);
        }
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      if constexpr (order % 2 == 0 || dimension != dim)
        {
          return (*this);
        }
      else
        {
          return Rational<-1, 1>{} * (*this);
        }
    }


    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      if constexpr (dimension == dim)
        {
          return (fact ^ Intg<order>{}) * (*this);
        }
      else
        {
          return (*this);
        }
    }

  };

  template <class T2, indexer order, indexer dim,
            typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const T2& other, const Monomial<order, dim>& mono)
  {
    return other * Monomial<-order, dim>{};
  }
  
  template <indexer order, indexer dim, indexer val>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator ^ (const Monomial<order, dim>& mono, const Intg<val>& exp)
  {
    return Monomial<order*val, dim>{};
  }

  template <indexer order, indexer dim>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator + (const Monomial<order, dim> &x1, const Monomial<order, dim> &x2)
  {
    return Rational<2, 1>{} * Monomial<order, dim>{};
  }
  template <indexer order, indexer dim>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const Monomial<order, dim> &x1, const Monomial<order, dim> &x2)
  {
    return Zero{};
  }
  template <indexer dim, indexer order_1, indexer order_2>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator * (const Monomial<order_1, dim> &x1, const Monomial<order_2, dim> &x2)
  {
    return Monomial<order_1 + order_2, dim>{};
  }
  template <indexer dim, indexer order_1, indexer order_2>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const Monomial<order_1, dim> &x1, const Monomial<order_2, dim> &x2)
  {
    return Monomial<order_1 - order_2, dim>{};
  }
    
}

#endif
