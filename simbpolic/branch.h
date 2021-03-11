#ifndef SIMBPOLIC_BRANCH
#define SIMBPOLIC_BRANCH
namespace Simbpolic
{

  template <class A, class B, indexer dim, class Cut> struct branch_function :
  public func_holder<A, B, Cut>, public SymBase, public std::conditional_t<is_exact<Cut>, SymExactBranched, SymBranched>
  {
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    
    using func_holder<A, B, Cut>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A, B, Cut>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A, B, Cut>::template get<1>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto cut() const
    {
      return func_holder<A, B, Cut>::template get<2>();
    }
    
    template <class dummy = A, typename std::enable_if_t<std::is_same_v<dummy, B> && is_exact<dummy>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator dummy () const
    {
      return dummy{};
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dimension>() || B::template has_dimension<dimension>() || (dimension == dim);
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant();
    }
        
    static constexpr indexer min_dimension = (dim < A::min_dimension && dim < B::min_dimension ? dim :
                                                (A::min_dimension < B::min_dimension ? A::min_dimension : B::min_dimension));
    static constexpr indexer max_dimension = (dim > A::max_dimension && dim > B::max_dimension ? dim :
                                                (A::max_dimension > B::max_dimension ? A::max_dimension : B::max_dimension));
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dimension>();
      constexpr indexer c2 = B::template integral_complexity<dimension>();
      return (c1 > c2 ? c1 : c2);
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return (dimension != dim) || (std::is_same_v<A, B> && is_exact<A>);
    }
    
    friend std::ostream& operator << (std::ostream &s, const branch_function& z)
    {
        s << "{ " << z.f1() << " , x_{" << dim << "} < " << z.cut() << " ; "
          << z.f2() << " , " << z.cut() << " < x_{" << dim << "} }";
        return s;
    }
         
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      if constexpr (std::is_convertible_v<A, Zero> && std::is_convertible_v<B, Zero>)
        {
          return Zero{};
        }
      else if constexpr (std::is_same_v<A, B> && is_exact<A>)
        {
          return A{} * Monomial<1, dimension>{};
        }
      else if constexpr (has_dimension<dimension>())
        {
          const auto prim_1 = f1().template primitive<dimension>();
          const auto prim_2 = f2().template primitive<dimension>();
          const auto second = prim_2 - prim_2.template evaluate_along_dim<dim>(cut())
                                     + prim_1.template evaluate_along_dim<dim>(cut());
          //So that the integration can still be performed by the difference of the primitives.
          
          return branch_function<decltype(prim_1), decltype(second), dim, Cut>{prim_1, second, cut()};
              
        }
      else
        {
          return (*this) * Monomial<1, dimension>{};
        }
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      if constexpr (is_constant())
        {
          return Zero{};
        }
      else if constexpr (has_dimension<dimension>())
        {
          const auto deriv_1 = f1().template derivative<dimension>();
          const auto deriv_2 = f2().template derivative<dimension>();
          return branch_function<decltype(deriv_1), decltype(deriv_2), dim, Cut>{deriv_1, deriv_2, cut()};
        }
      else
        {
          return Zero{};
        }
    }
    
    private:
    
    template <class Val>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate (const Val& val) const
    {
      if constexpr (is_exact<Val> && is_exact<Cut>)
        {
          if constexpr (Val{} < Cut{})
            {
              return f1()(val);
            }
          else if constexpr (Val{} > Cut{})
            {
              return f2()(val);
            }
          else
            {
              return (f1()(val) + f2()(val))*Rational<1,2>{};
              //The usual extension...
            }
        }
      else if constexpr (is_stored<Val>)
        {
          return stored_conditional(val, f1(), cut(), f2());
        }
      else
        {
          if (val < Type(cut()))
            {
              return Constant{Type(f1()(val))};
            }
          else if (val > Type(cut()))
            {
              return Constant{Type(f2()(val))};
            }
          else
            {
              return Constant{(Type(f1()(val)) + Type(f2()(val)))/Type(2)};
              //The usual extension...
            }
        }
    }
    
    template <class Store, class Val>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_evaluate (const Store& store, const Val& val) const
    {
      if constexpr (is_exact<Val> && is_exact<Cut>)
        {
          if constexpr (Val{} < Cut{})
            {
              return f1()(store, val);
            }
          else if constexpr (Val{} > Cut{})
            {
              return f2()(store, val);
            }
          else
            {
              return (f1()(store, val) + f2()(store, val))*Rational<1,2>{};
              //The usual extension...
            }
        }
      else
        {
          if (Type(val(store)) < Type(cut()(store)))
            {
              return Constant{Type(f1()(store, val))};
            }
          else if (Type(val(store)) < Type(cut()(store)))
            {
              return Constant{Type(f2()(store, val))};
            }
          else
            {
              return Constant{(Type(f1()(store, val)) + Type(f2()(store, val)))/Type(2)};
              //The usual extension...
            }
        }
    }
    
    public:
    
    template <indexer idx>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto decide() const
    {
      return (*this);
    }
    
    template <indexer idx, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto decide(const Arg& arg) const
    {
      if constexpr (idx == dim && (is_numeric<Arg> || is_stored<Arg>))
        {
          return evaluate(arg);
        }
      else
        {
          return (*this);
        }
    }
    
    template <indexer idx, class First, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto decide (const First& f, const Args& ... args) const
    {
      if constexpr (idx > dim)
        {
          return (*this);
        }
      else if constexpr (idx == dim)
        {
          return decide<idx>(f);
        }
      else
        {
          return decide<idx + 1>(args...);
        }
    }
    
    template <indexer idx, class Store>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_decide(const Store& store) const
    {
      const auto new_cut = cut()(store);
      return branch_function<A, B, dim, decltype(new_cut)>{f1(), f2(), new_cut};
    }
    
    template <indexer idx, class Store, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_decide(const Store& store, const Arg& arg) const
    {
      if constexpr (idx == dim && (is_numeric<Arg> || is_stored<Arg>))
        {
          return stored_evaluate(store, arg);
        }
      else
        {
          const auto new_cut = cut()(store);
          return branch_function<A, B, dim, decltype(new_cut)>{f1(), f2(), new_cut};
        }
    }
    
    template <indexer idx, class Store, class First, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_decide(const Store& store, const First& f, const Args& ... args) const
    {
      if constexpr (idx > dim)
        {
          const auto new_cut = cut()(store);
          return branch_function<A, B, dim, decltype(new_cut)>{f1(), f2(), new_cut};
        }
      else if constexpr (idx == dim)
        {
          return stored_decide<idx>(store, f);
        }
      else
        {
          return stored_decide<idx + 1>(store, args...);
        }
    }
    
    template <class Arg, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Arg& first, const Args& ... args) const
    {
      if constexpr (is_store<Arg>)
        {
          const auto result1 = f1()(args...);
          const auto result2 = f2()(args...);
          const auto ret = branch_function<decltype(result1), decltype(result2), dim, Cut>{result1, result2, cut()};
          
          return ret.template stored_decide<1>(first, args...);
        }
      else
        {
          const auto result1 = f1()(first, args...);
          const auto result2 = f2()(first, args...);
          const auto ret = branch_function<decltype(result1), decltype(result2), dim, Cut>{result1, result2, cut()};
          
          return ret.template decide<1>(first, args...);
        }
    }
    
    template <indexer dimension, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim (const Arg& val) const
    {
      const auto result1 = f1().template evaluate_along_dim<dimension>(val);
      const auto result2 = f2().template evaluate_along_dim<dimension>(val);
      
      const auto ret = branch_function<decltype(result1), decltype(result2), dim, Cut>{result1, result2, cut()};
      
      return ret.template decide<dimension>(val);
      
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      const auto g_1 = Simbpolic::change_dim(x, y, f1());
      const auto g_2 = Simbpolic::change_dim(x, y, f2());
      return branch_function<decltype(g_1), decltype(g_2), (dim == from ? to : dim), Cut>{g_1, g_2, cut()};
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      if constexpr (dimension == dim)
        {
          const auto g_1 = Simbpolic::offset(x, off, f1());
          const auto g_2 = Simbpolic::offset(x, off, f2());
          const auto new_cut = cut() - off;
          return branch_function<decltype(g_1), decltype(g_2), dim, decltype(new_cut)>{g_1, g_2, new_cut};
        }
      else
        {
          const auto g_1 = Simbpolic::offset(x, off, f1());
          const auto g_2 = Simbpolic::offset(x, off, f2());
          return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, cut()};
        }
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      const auto g_1 = Simbpolic::reverse(f1());
      const auto g_2 = Simbpolic::reverse(f2());
      if constexpr (dimension == dim)
        {
          const auto new_cut = -cut();
          return branch_function<decltype(g_2), decltype(g_1), dim, decltype(new_cut)>{g_2, g_1, new_cut};
        }
      else
        {
          return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, cut()};
        }
    }


    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      if constexpr (dimension == dim)
        {
          const auto g_1 = Simbpolic::deform(x, fact, f1());
          const auto g_2 = Simbpolic::deform(x, fact, f2());
          const auto new_cut = cut() * fact;
          return branch_function<decltype(g_1), decltype(g_2), dim, decltype(new_cut)>{g_1, g_2, new_cut};
        }
      else
        {
          const auto g_1 = Simbpolic::deform(x, fact, f1());
          const auto g_2 = Simbpolic::deform(x, fact, f2());
          return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, cut()};
        }
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      const auto g_1 = Simbpolic::distribute<recurse_count-1>(f1());
      const auto g_2 = Simbpolic::distribute<recurse_count-1>(f2());
      return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, cut()};
    }
    
  };


  #define SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS(OP)             \
  template <indexer dim, class A, class B, class Cut, class F1, class F2, class Other_cut, \
            typename std::enable_if_t<is_exact<Cut> && is_exact<Other_cut>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP ( const branch_function<A, B, dim, Cut>&w, \
                                                             const branch_function<F1, F2, dim, Other_cut> &z) \
  {                                                             \
    if constexpr (Cut{} < Other_cut{})                          \
      {                                                         \
        return branched(Var<dim>{}, w.f1() OP z.f1(), Cut{}, w.f2() OP z.f1(), Other_cut{}, w.f2() OP z.f2()); \
      }                                                         \
    else if constexpr (Cut{} > Other_cut{})                     \
      {                                                         \
        return branched(Var<dim>{}, w.f1() OP z.f1(), Other_cut{}, w.f1() OP z.f2(), Cut{}, w.f2() OP z.f2()); \
      }                                                         \
    else                                                        \
      {                                                         \
        const auto first = w.f1() OP z.f1();                          \
        const auto second = w.f2() OP z.f2();                         \
        return branch_function<decltype(first), decltype(second), dim, Cut>{first, second, Cut{}}; \
      }                                                         \
  }                                                             \

  SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS(+);
  SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS(-);
  SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS(*);
  SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS(/);

  #undef SIMBPOLIC_EXACT_BRANCH_SELF_OPERATORS

  #define SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS(OP)                          \
  template <indexer dim, class A, class B, class Cut, class F1, class F2, class F3, class LowerCut, class UpperCut, \
            typename std::enable_if_t<is_exact<Cut> && is_exact<LowerCut> && is_exact<UpperCut>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP ( const branch_function<A, B, dim, Cut>&w, \
                                                             const interval_function<F1, F2, F3, dim, LowerCut, UpperCut> &z) \
  {                                                                              \
    if constexpr (Cut{} < LowerCut{})                                            \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), Cut{}, w.f2() OP z.f1(), LowerCut{}, w.f2() OP z.f2(), UpperCut{}, w.f2() OP z.f3() ); \
      }                                                                          \
    else if constexpr (Cut{} == LowerCut{})                                      \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), Cut{}, w.f2() OP z.f2(), UpperCut{}, w.f2() OP z.f3() ); \
      }                                                                          \
    else if constexpr (Cut{} < UpperCut{})                                       \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f1() OP z.f2(), Cut{}, w.f2() OP z.f2(), UpperCut{}, w.f2() OP z.f3() ); \
      }                                                                          \
    else if constexpr (Cut{} == UpperCut{})                                      \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f1() OP z.f2(), Cut{}, w.f2() OP z.f3() ); \
      }                                                                          \
    else                                                                         \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f1() OP z.f2(), UpperCut{}, w.f1() OP z.f3(), Cut{}, w.f2() OP z.f3() ); \
      }                                                                          \
  }                                                                              \


  SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS(+);
  SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS(-);
  SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS(*);
  SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS(/);

  #undef SIMBPOLIC_EXACT_BRANCH_INTERVAL_OPERATORS


  #define SIMBPOLIC_BRANCH_OTHER_OPERATORS(OP)          \
  template<class F, class A, class B, indexer dim, class Cut, \
  typename std::enable_if_t<!is_branched<F> && !is_exceptional<F>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const F& f, const branch_function<A, B, dim, Cut>& branch) \
  {                                                           \
    const auto g_1 = f OP branch.f1();                          \
    const auto g_2 = f OP branch.f2();                          \
    return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, branch.cut()}; \
  }                                                           \
  template<class F, class A, class B, indexer dim, class Cut, \
  typename std::enable_if_t<!is_branched<F> && !is_exceptional<F>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const branch_function<A, B, dim, Cut>& branch, const F& f) \
  {                                                           \
    const auto g_1 = branch.f1() OP f;                          \
    const auto g_2 = branch.f2() OP f;                          \
    return branch_function<decltype(g_1), decltype(g_2), dim, Cut>{g_1, g_2, branch.cut()}; \
  }                                                           \

  SIMBPOLIC_BRANCH_OTHER_OPERATORS(+);
  SIMBPOLIC_BRANCH_OTHER_OPERATORS(-);
  SIMBPOLIC_BRANCH_OTHER_OPERATORS(*);
  SIMBPOLIC_BRANCH_OTHER_OPERATORS(/);
  SIMBPOLIC_BRANCH_OTHER_OPERATORS(^);
  
  #undef SIMBPOLIC_BRANCH_OTHER_OPERATORS
  
}
#endif
