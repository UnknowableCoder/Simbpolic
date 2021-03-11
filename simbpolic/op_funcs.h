#ifndef SIMBPOLIC_OP_FUNCS
#define SIMBPOLIC_OP_FUNCS
namespace Simbpolic
{
  
  struct mult_distributable {};

  template <class A, class B> struct func_add : public func_holder<A, B>, public SymBase, public mult_distributable, public SymOpFunc
  { 
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    
    using func_holder<A, B>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A,B>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A,B>::template get<1>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dim>() || B::template has_dimension<dim>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant();
    }
        
    static constexpr indexer min_dimension = (A::min_dimension < B::min_dimension ? A::min_dimension : B::min_dimension);
    static constexpr indexer max_dimension = (A::max_dimension > B::max_dimension ? A::max_dimension : B::max_dimension);
    
    friend std::ostream& operator << (std::ostream &s, const func_add& z)
    {
        s << "( " << z.f1() << " ) + ( " << z.f2() << " )";
        return s;
    }

    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dim>();
      constexpr indexer c2 = B::template integral_complexity<dim>();
      return (c1 > c2 ? c1 : c2);
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return A::template is_continuous<dimension>() && B::template is_continuous<dimension>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return f1().template primitive<dim>() + f2().template primitive<dim>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return f1().template derivative<dim>() + f2().template derivative<dim>();
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return f1().template evaluate_along_dim<dim>(args...) + f2().template evaluate_along_dim<dim>(args...);
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return f1()(args...) + f2()(args...);
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(f1()) + T1(f2());
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      return Simbpolic::change_dim(x, y, f1()) + Simbpolic::change_dim(x, y, f2());
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      return Simbpolic::offset(x, off, f1()) + Simbpolic::offset(x, off, f2());
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      return Simbpolic::reverse(x, f1()) + Simbpolic::reverse(x, f2());
    }

    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      return Simbpolic::deform(x, fact, f1()) + Simbpolic::deform(x, fact, f2());
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      return Simbpolic::distribute<recurse_count-1>(f1()) + Simbpolic::distribute<recurse_count-1>(f2());
    }
        
    template <class Op1, class Op2>
    SIMBPOLIC_CUDA_HOS_DEV static inline constexpr auto substitute(const Op1& left, const Op2& right)
    {
      return left + right;
    }
    
  };

  template <class A, class B> struct func_sub : public func_holder<A, B>, public SymBase, public mult_distributable, public SymOpFunc
  {
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    
    using func_holder<A, B>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A,B>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A,B>::template get<1>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dim>() || B::template has_dimension<dim>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant();
    }
    
    static constexpr indexer min_dimension = (A::min_dimension < B::min_dimension ? A::min_dimension : B::min_dimension);
    static constexpr indexer max_dimension = (A::max_dimension > B::max_dimension ? A::max_dimension : B::max_dimension);
    
    friend std::ostream& operator << (std::ostream &s, const func_sub& z)
    {
        s << "( " << z.f1() << " ) - ( " << z.f2() << " )";
        return s;
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dim>();
      constexpr indexer c2 = B::template integral_complexity<dim>();
      return (c1 > c2 ? c1 : c2);
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return A::template is_continuous<dimension>() && B::template is_continuous<dimension>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return f1().template primitive<dim>() - f2().template primitive<dim>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return f1().template derivative<dim>() - f2().template derivative<dim>();
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return f1().template evaluate_along_dim<dim>(args...) - f2().template evaluate_along_dim<dim>(args...);
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return f1()(args...) - f2()(args...);
    }
      
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(f1()) - T1(f2());
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      return Simbpolic::change_dim(x, y, f1()) - Simbpolic::change_dim(x, y, f2());
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      return Simbpolic::offset(x, off, f1()) - Simbpolic::offset(x, off, f2());
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      return Simbpolic::reverse(x, f1()) - Simbpolic::reverse(x, f2());
    }

    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      return Simbpolic::deform(x, fact, f1()) - Simbpolic::deform(x, fact, f2());
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      return Simbpolic::distribute<recurse_count-1>(f1()) - Simbpolic::distribute<recurse_count-1>(f2());
    }
        
    template <class Op1, class Op2>
    SIMBPOLIC_CUDA_HOS_DEV static inline constexpr auto substitute(const Op1& left, const Op2& right)
    {
      return left - right;
    }
  };

  template <class A, class B> struct func_mul : public func_holder<A, B>, public SymBase, public SymOpFunc
  {
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    
    using func_holder<A, B>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A,B>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A,B>::template get<1>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dim>() || B::template has_dimension<dim>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant();
    }
        
    static constexpr indexer min_dimension = (A::min_dimension < B::min_dimension ? A::min_dimension : B::min_dimension);
    static constexpr indexer max_dimension = (A::max_dimension > B::max_dimension ? A::max_dimension : B::max_dimension);
    
    friend std::ostream& operator << (std::ostream &s, const func_mul& z)
    {
        s << "( " << z.f1() << " ) * ( " << z.f2() << " )";
        return s;
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dim>();
      constexpr indexer c2 = B::template integral_complexity<dim>();
      return 4*(c1 + c2);
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return A::template is_continuous<dimension>() && B::template is_continuous<dimension>();
    }
    
    private:
    /*
    template <class Func>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr inline auto continuity_correction(const Func& f)
    {
      return f;
    }
    
    //Only relevant when integrating along dim!
    template <class A, class B, class Cut, indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr inline auto continuity_correction(const branch_function<A, B, Cut>& f)
    {
      
    }
    */
    public:
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    //This currently is far from being optimized!
    //May possibly lead to infinite compilation times!
    //Beware, if compiling is taking too much time,
    //perhaps switching the order of integration might be sensible!
    {
      if constexpr (A::template has_dimension<dim>() && B::template has_dimension<dim>())
        {
          const auto prim1 = f1().template primitive<dim>();
          const auto prim2 = f2().template primitive<dim>();
          const auto deriv1 = f1().template derivative<dim>();
          const auto deriv2 = f2().template derivative<dim>();
          
          const auto first_1 = prim1 * f2();
          const auto first_2 = f1() * prim2;
          
          const auto second_1 = prim1 * deriv2;
          const auto second_2 = deriv1 * prim2;
          
          constexpr indexer c1 = decltype(second_1)::template integral_complexity<dim>();
          constexpr indexer c2 = decltype(second_2)::template integral_complexity<dim>();
          
          if constexpr ((c1 <= c2 && decltype(deriv1)::template is_continuous<dim>()
                                  && decltype(deriv2)::template is_continuous<dim>()) ||
                        (!decltype(deriv1)::template is_continuous<dim>() &&
                          decltype(deriv2)::template is_continuous<dim>()    ) )
            {
              const auto temp_prim = second_1.template primitive<dim>();
              const auto ret = first_1 - temp_prim;
              return ret;
            }
          else if constexpr (decltype(deriv1)::template is_continuous<dim>())
            {
              const auto temp_prim = second_2.template primitive<dim>();
              const auto ret = first_2 - temp_prim;
              return ret;
            }
          else
            {
              return Zero{};
              //static_assert(sizeof(char) == 0, "We do not yet support integration by parts with two branched functions, sorry.");
            }
        }
      else
        {
          if constexpr (A::template has_dimension<dim>())
            {
              const auto temp = f1().template primitive<dim>();
              return temp * f2();
            }
          else if constexpr (B::template has_dimension<dim>())
            {
              const auto temp = f2().template primitive<dim>();
              return f1() * temp;
            }
          else
            {
              return Monomial<1, dim>{} * (*this);
            }
        }
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return (f1().template derivative<dim>()) * f2() + f1() * (f2().template derivative<dim>());
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return f1().template evaluate_along_dim<dim>(args...) * f2().template evaluate_along_dim<dim>(args...);
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return f1()(args...) * f2()(args...);
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(f1()) * T1(f2());
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      return Simbpolic::change_dim(x, y, f1()) * Simbpolic::change_dim(x, y, f2());
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      return Simbpolic::offset(x, off, f1()) * Simbpolic::offset(x, off, f2());
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      return Simbpolic::reverse(x, f1()) * Simbpolic::reverse(x, f2());
    }

    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      return Simbpolic::deform(x, fact, f1()) * Simbpolic::deform(x, fact, f2());
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      if constexpr (std::is_base_of_v<mult_distributable, A> && std::is_base_of_v<mult_distributable, B>)
        {
          const auto a = Simbpolic::distribute<recurse_count - 1>(f1().f1());
          const auto b = Simbpolic::distribute<recurse_count - 1>(f1().f2());
          const auto c = Simbpolic::distribute<recurse_count - 1>(f2().f1());
          const auto d = Simbpolic::distribute<recurse_count - 1>(f2().f2());
          
          const auto ret = f1().substitute(f2().substitute(a*c, a*d), f2().substitute(b*c, b*d));
          
          return Simbpolic::distribute<recurse_count - 1>(ret);
        }
      else if constexpr (std::is_base_of_v<mult_distributable, A>)
        {
          const auto a = Simbpolic::distribute<recurse_count - 1>(f1().f1());
          const auto b = Simbpolic::distribute<recurse_count - 1>(f1().f2());
          
          const auto c = Simbpolic::distribute<recurse_count - 1>(f2());
          
          const auto ret = f1().substitute(a * c, b * c);
          
          return Simbpolic::distribute<recurse_count - 1>(ret);
        }
      else if constexpr (std::is_base_of_v<mult_distributable, B>)
        {
          const auto a = Simbpolic::distribute<recurse_count - 1>(f1());
          
          const auto c = Simbpolic::distribute<recurse_count - 1>(f2().f1());
          const auto d = Simbpolic::distribute<recurse_count - 1>(f2().f2());
          
          
          const auto ret = f2().substitute(a * c, a * d);
          
          return Simbpolic::distribute<recurse_count - 1>(ret);
        }
      else
        {
          return Simbpolic::distribute<recurse_count - 1>(f1()) * Simbpolic::distribute<recurse_count - 1>(f2());
        }
    }
    
    template <class Op1, class Op2>
    SIMBPOLIC_CUDA_HOS_DEV static inline constexpr auto substitute(const Op1& left, const Op2& right)
    {
      return left * right;
    }
  };


  template <class A, class B> struct func_div : public func_holder<A, B>, public SymBase, public SymOpFunc
  {
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    
    using func_holder<A, B>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A,B>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A,B>::template get<1>();
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dim>() || B::template has_dimension<dim>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant();
    }
    
    static constexpr indexer min_dimension = (A::min_dimension < B::min_dimension ? A::min_dimension : B::min_dimension);
    static constexpr indexer max_dimension = (A::max_dimension > B::max_dimension ? A::max_dimension : B::max_dimension);
    
    friend std::ostream& operator << (std::ostream &s, const func_div& z)
    {
        s << "\\frac{ " << z.f1() << " }{ " << z.f2() << " }";
        return s;
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dim>();
      constexpr indexer c2 = B::template integral_complexity<dim>();
      return 4 * (c1+c2);
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return A::template is_continuous<dimension>() && B::template is_continuous<dimension>();
    }
        
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      if constexpr (B::is_constant())
        {
          return f1().template primitive<dim>() / f2();
        }
      else
        {
          static_assert(dim == 0, "Integration of reciprocals not yet supported, sorry.");
        }
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return (f1().template derivative<dim>()) / f2() + f1() * (f2().template derivative<dim>())/(f2() * f2());
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return f1().template evaluate_along_dim<dim>(args...) / f2().template evaluate_along_dim<dim>(args...);
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return f1()(args...) / f2()(args...);
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(f1()) / T1(f2());
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      return Simbpolic::change_dim(x, y, f1()) / Simbpolic::change_dim(x, y, f2());
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      return Simbpolic::offset(x, off, f1()) / Simbpolic::offset(x, off, f2());
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      return Simbpolic::reverse(x, f1()) / Simbpolic::reverse(x, f2());
    }

    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      return Simbpolic::deform(x, fact, f1()) / Simbpolic::deform(x, fact, f2());
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      if constexpr (std::is_base_of_v<mult_distributable, A>)
        {
          const auto dist_denom = Simbpolic::distribute<recurse_count-1>(f2());
          return f1().substitute(Simbpolic::distribute<recurse_count-1>(f1().f1())/dist_denom, Simbpolic::distribute<recurse_count-1>(f1().f2())/dist_denom);
        }
      else
        {
          return Simbpolic::distribute<recurse_count-1>(f1()) / Simbpolic::distribute<recurse_count-1>(f2());
        }
    }
    
    template <class Op1, class Op2>
    SIMBPOLIC_CUDA_HOS_DEV static inline constexpr auto substitute(const Op1& left, const Op2& right)
    {
      return left / right;
    }
  };
}
#endif