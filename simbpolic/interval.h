#ifndef SIMBPOLIC_INTERVAL
#define SIMBPOLIC_INTERVAL
namespace Simbpolic
{

  template <class A, class B, class C, indexer dim, class LowerCut, class UpperCut> struct interval_function :
  public func_holder<A, B, C, LowerCut, UpperCut>,
  public SymBase,
  public std::conditional_t<is_exact<LowerCut> && is_exact<UpperCut>, SymExactBranched, SymBranched>
  {
    static_assert(is_symbolic<A>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<B>, "Should be called with symbolic functions!");
    static_assert(is_symbolic<C>, "Should be called with symbolic functions!");
    
    using func_holder<A, B, C, LowerCut, UpperCut>::func_holder;
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f1() const
    {
      return func_holder<A, B, C, LowerCut, UpperCut>::template get<0>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f2() const
    {
      return func_holder<A, B, C, LowerCut, UpperCut>::template get<1>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto f3() const
    {
      return func_holder<A, B, C, LowerCut, UpperCut>::template get<2>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto lower_cut() const
    {
      return func_holder<A, B, C, LowerCut, UpperCut>::template get<3>();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto upper_cut() const
    {
      return func_holder<A, B, C, LowerCut, UpperCut>::template get<4>();
    }
    
    
    template <class dummy = A, typename std::enable_if_t<std::is_same_v<dummy, B> && std::is_same_v<dummy, C> && is_exact<dummy>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator dummy () const
    {
      return dummy{};
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV static constexpr bool has_dimension()
    {
      return A::template has_dimension<dimension>() || B::template has_dimension<dimension>() ||
             C::template has_dimension<dimension>() || (dimension == dim);
    }
    
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_constant()
    {
      return A::is_constant() && B::is_constant() && C::is_constant();
    }
        
    static constexpr indexer min_dimension = (dim < A::min_dimension && dim < B::min_dimension && dim < C::min_dimension ? dim :
                                                (A::min_dimension < B::min_dimension ? 
                                                                                       ( A::min_dimension < C::min_dimension ? 
                                                                                         A::min_dimension : C::min_dimension   )
                                                                                      :
                                                                                       ( B::min_dimension < C::min_dimension ? 
                                                                                         B::min_dimension : C::min_dimension   )
                                                 )
                                              );
    static constexpr indexer max_dimension = (dim > A::max_dimension && dim > B::max_dimension && dim > C::max_dimension ? dim :
                                                (A::max_dimension > B::max_dimension ? 
                                                                                       ( A::max_dimension > C::max_dimension ? 
                                                                                         A::max_dimension : C::max_dimension   )
                                                                                      :
                                                                                       ( B::max_dimension > C::max_dimension ? 
                                                                                         B::max_dimension : C::max_dimension   )
                                                 )
                                              );
    


    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer integral_complexity()
    {
      constexpr indexer c1 = A::template integral_complexity<dimension>();
      constexpr indexer c2 = B::template integral_complexity<dimension>();
      constexpr indexer c3 = C::template integral_complexity<dimension>();
      return (c1 > c2 ? (c1 > c3 ? c1 : c3) : (c2 > c3 ? c2 : c3));
    }
    
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool is_continuous()
    {
      return (dimension != dim) || (std::is_same_v<A, B> && std::is_same_v<B, C> && is_exact<A>);
    }
    
    friend std::ostream& operator << (std::ostream &s, const interval_function& z)
    {
        s << "{ " << z.f1() << " , x_{" << dim << "} < " << z.lower_cut() << " ; "
                  << z.f2() << " , " << z.lower_cut() << " < x_{" << dim << "} < " << z.upper_cut() << " ; "
                  << z.f3() << " , " << z.upper_cut() << " < x_{" << dim << "} }";
        return s;
    }
         
    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      if constexpr (std::is_convertible_v<A, Zero> && std::is_convertible_v<B, Zero> && std::is_convertible_v<C, Zero>)
        {
          return Zero{};
        }
      else if constexpr (std::is_same_v<A, B> && std::is_same_v<B, C> && is_exact<A>)
        {
          return A{} * Monomial<1, dimension>{};
        }
      else if constexpr (has_dimension<dimension>())
        {
          const auto prim_1 = f1().template primitive<dimension>();
          const auto prim_2 = f2().template primitive<dimension>();
          const auto prim_3 = f3().template primitive<dimension>();
          const auto second = prim_2 - prim_2.template evaluate_along_dim<dim>(lower_cut())
                                     + prim_1.template evaluate_along_dim<dim>(lower_cut());
          
          const auto third = prim_3 - prim_3.template evaluate_along_dim<dim>(upper_cut())
                                    + second.template evaluate_along_dim<dim>(upper_cut());
          //So that the integration can still be performed by the difference of the primitives.
          
          return interval_function<decltype(prim_1), decltype(second), decltype(third), dim, LowerCut, UpperCut>
                          {prim_1, second, third, lower_cut(), upper_cut()};
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
          const auto deriv_3 = f3().template derivative<dimension>();
          return interval_function<decltype(deriv_1), decltype(deriv_2), decltype(deriv_3), dim, LowerCut, UpperCut>
                          {deriv_1, deriv_2, deriv_3, lower_cut(), upper_cut()};
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
      if constexpr(is_exact<Val> && is_exact<LowerCut> && is_exact<UpperCut>)
        {
          if constexpr (Val{} < LowerCut{})
            {
              return f1()(val);
            }
          else if constexpr (Val{} == LowerCut{})
            {
              return (f1() + f2())*Rational<1,2>{};
            }
          else if constexpr (Val{} < UpperCut{})
            {
              return f2()(val);
            }
          else if constexpr (Val{} == UpperCut{})
            {
              return (f2() + f3())*Rational<1,2>{};
            }
          else
            {
              return f3()(val);
            }
        }
      else if constexpr (is_stored<Val>)
        {
          return stored_conditional(val, f1(), lower_cut(), f2(), upper_cut(), f3());
        }
      else
        {
          if (Type(val) < Type(lower_cut()))
            {
              return Constant{Type(f1()(val))};
            }
          else if (Type(val) == Type(lower_cut()))
            {
              return Constant{(Type(f1()(val)) + Type(f2()(val)))/Type(2)};
            }
          else if (Type(val) < Type(upper_cut()))
            {
              return Constant{Type(f2()(val))};
            }
          else if (Type(val) == Type(upper_cut()))
            {
              return Constant{(Type(f2()(val)) + Type(f3()(val)))/Type(2)};
              //The usual extension...
            }
          else
            {
              return Constant{Type(f3()(val))};
            }
        }
    }
    
    template <class Store,class Val>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_evaluate (const Store& store, const Val& val) const
    {
      if constexpr(is_exact<Val> && is_exact<LowerCut> && is_exact<UpperCut>)
        {
          if constexpr (Val{} < LowerCut{})
            {
              return f1()(store, val);
            }
          else if constexpr (Val{} == LowerCut{})
            {
              return (f1()(store, val) + f2()(store, val))*Rational<1,2>{};
            }
          else if constexpr (Val{} < UpperCut{})
            {
              return f2()(store, val);
            }
          else if constexpr (Val{} == UpperCut{})
            {
              return (f2()(store, val) + f3()(store, val))*Rational<1,2>{};
            }
          else
            {
              return f3()(store, val);
            }
        }
      else
        {
          if (Type(val(store)) < Type(lower_cut()(store)))
            {
              return Constant{Type(f1()(store, val))};
            }
          else if (Type(val(store)) == Type(lower_cut()(store)))
            {
              return Constant{(Type(f1()(store, val)) + Type(f2()(store, val)))/Type(2)};
            }
          else if (Type(val(store)) < Type(upper_cut()(store)))
            {
              return Constant{Type(f2()(store, val))};
            }
          else if (Type(val(store)) == Type(upper_cut()(store)))
            {
              return Constant{(Type(f2()(store, val)) + Type(f3()(store, val)))/Type(2)};
              //The usual extension...
            }
          else
            {
              return Constant{Type(f3()(store, val))};
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
      const auto new_lower = lower_cut()(store);
      const auto new_upper = upper_cut()(store);
      return interval_function<A, B, C, dim, decltype(new_lower), decltype(new_upper)>{f1(), f2(), f3(), new_lower, new_upper};
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
          const auto new_lower = lower_cut()(store);
          const auto new_upper = upper_cut()(store);
          return interval_function<A, B, C, dim, decltype(new_lower), decltype(new_upper)>{f1(), f2(), f3(), new_lower, new_upper};
        }
    }
    
    template <indexer idx, class Store, class First, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto stored_decide(const Store& store, const First& f, const Args& ... args) const
    {
      if constexpr (idx > dim)
        {
          const auto new_lower = lower_cut()(store);
          const auto new_upper = upper_cut()(store);
          return interval_function<A, B, C, dim, decltype(new_lower), decltype(new_upper)>{f1(), f2(), f3(), new_lower, new_upper};
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
          const auto result3 = f3()(args...);
          const auto ret = interval_function<decltype(result1), decltype(result2), decltype(result3), dim, LowerCut, UpperCut>{result1, result2, result3, lower_cut(), upper_cut()};
          
          return ret.template stored_decide<1>(first, args...);
        }
      else
        {
          const auto result1 = f1()(first, args...);
          const auto result2 = f2()(first, args...);
          const auto result3 = f3()(args...);
          const auto ret = interval_function<decltype(result1), decltype(result2), decltype(result3), dim, LowerCut, UpperCut>{result1, result2, result3, lower_cut(), upper_cut()};
          
          return ret.template decide<1>(first, args...);
        }
    }
    
    template <indexer dimension, class Arg>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim (const Arg& val) const
    {
      const auto result1 = f1().template evaluate_along_dim<dimension>(val);
      const auto result2 = f2().template evaluate_along_dim<dimension>(val);
      const auto result3 = f3().template evaluate_along_dim<dimension>(val);
      
      const auto ret = interval_function<decltype(result1), decltype(result2), decltype(result3), dim, LowerCut, UpperCut>
                                {result1, result2, result3, lower_cut(), upper_cut()};
      
      return ret.template decide<dimension>(val);
    }
    
    template <indexer from, indexer to>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto change_dim (const Var<from> &x, const Var<to> &y) const
    {
      const auto g_1 = Simbpolic::change_dim(x, y, f1());
      const auto g_2 = Simbpolic::change_dim(x, y, f2());
      const auto g_3 = Simbpolic::change_dim(x, y, f3());
      return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), (dim == from ? to : dim), LowerCut, UpperCut>{g_1, g_2, g_3, lower_cut(), upper_cut()};
    }

    template <indexer dimension, class Off>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto offset(const Var<dimension> &x, const Off& off) const
    {
      if constexpr (dimension == dim)
        {
          const auto g_1 = Simbpolic::offset(x, off, f1());
          const auto g_2 = Simbpolic::offset(x, off, f2());
          const auto g_3 = Simbpolic::offset(x, off, f3());
          const auto new_low = lower_cut() - off;
          const auto new_up = upper_cut() - off;
          return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, decltype(new_low), decltype(new_up)>{g_1, g_2, g_3, new_low, new_up};
        }
      else
        {
          const auto g_1 = Simbpolic::offset(x, off, f1());
          const auto g_2 = Simbpolic::offset(x, off, f2());
          const auto g_3 = Simbpolic::offset(x, off, f3());
          return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, lower_cut(), upper_cut()};
        }
    }

    template <indexer dimension>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto reverse(const Var<dimension> &x) const
    {
      const auto g_1 = Simbpolic::reverse(f1());
      const auto g_2 = Simbpolic::reverse(f2());
      const auto g_3 = Simbpolic::reverse(f3());
      if constexpr (dimension == dim)
        {
          const auto new_low = -upper_cut();
          const auto new_up = -lower_cut();
          return interval_function<decltype(g_3), decltype(g_2), decltype(g_1), dim, decltype(new_low), decltype(new_up)>{g_3, g_2, g_1, new_low, new_up};
        }
      else
        {
          return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, lower_cut(), upper_cut()};
        }
    }


    template <indexer dimension, class Val>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto deform(const Var<dimension> &x, const Val& fact) const
    {
      if constexpr (dimension == dim)
        {
          const auto g_1 = Simbpolic::deform(x, fact, f1());
          const auto g_2 = Simbpolic::deform(x, fact, f2());
          const auto g_3 = Simbpolic::deform(x, fact, f2());
          const auto new_low = lower_cut() * fact;
          const auto new_up = upper_cut() * fact;
          return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, decltype(new_low), decltype(new_up)>{g_1, g_2, g_3, new_low, new_up};
        }
      else
        {
          const auto g_1 = Simbpolic::deform(x, fact, f1());
          const auto g_2 = Simbpolic::deform(x, fact, f2());
          const auto g_3 = Simbpolic::deform(x, fact, f2());
          return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, lower_cut(), upper_cut()};
        }
    }

    template <indexer recurse_count>
    SIMBPOLIC_CUDA_HOS_DEV inline constexpr auto distribute() const
    {
      const auto g_1 = Simbpolic::distribute<recurse_count-1>(f1());
      const auto g_2 = Simbpolic::distribute<recurse_count-1>(f2());
      const auto g_3 = Simbpolic::distribute<recurse_count-1>(f3());
      return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, lower_cut(), upper_cut()};
    }
  };


  #define SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(OP)                                 \
  template <class A, class B, class C, indexer dim, class LowerCut, class UpperCut, \
            class F1, class F2, class F3, class Other_Lower, class Other_Upper, \
            typename std::enable_if_t<is_exact<LowerCut> && is_exact<UpperCut> && is_exact<Other_Lower> && is_exact<Other_Upper>>* = nullptr>  \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const interval_function<A, B, C, dim, LowerCut, UpperCut> &w, \
                                                            const interval_function<F1, F2, F3, dim, Other_Lower, Other_Upper> &z) \
  {                                                                              \
    if constexpr (LowerCut{} < Other_Lower{})                                    \
      {                                                                          \
        if constexpr (UpperCut{} < Other_Lower{})                                \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f1(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f3() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else if constexpr (UpperCut{} == Other_Lower{})                          \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f1(),                              \
                            UpperCut{},             /*Other_Lower{}*/            \
                                       w.f3() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else if constexpr (UpperCut{} < Other_Upper{})                           \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else if constexpr (UpperCut{} == Other_Upper{})                          \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},             /*Other_Upper{}*/            \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else                                                                     \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f2() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f2() OP z.f3(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
      }                                                                          \
    else if constexpr (LowerCut{} == Other_Lower{})                              \
      {                                                                          \
        if constexpr (UpperCut{} < Other_Upper{})                                \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},             /*Other_Lower{}*/            \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else if constexpr (UpperCut{} == Other_Upper{})                          \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},             /*Other_Lower{}*/            \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},             /*Other_Upper{}*/            \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else                                                                     \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                            LowerCut{},             /*Other_Lower{}*/            \
                                       w.f2() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f2() OP z.f3(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f3()                  );           \
                    }                                                            \
                }                                                                \
    else if constexpr (LowerCut{} < Other_Upper{})                               \
      {                                                                          \
        if constexpr (UpperCut{} < Other_Upper{})                                \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f1() OP z.f2(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
        else if constexpr (UpperCut{} == Other_Upper{})                          \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f1() OP z.f2(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f2(),                              \
                            UpperCut{},             /*Other_Upper{}*/            \
                                       w.f3() OP z.f3()                 );           \
          }                                                                      \
        else                                                                     \
          {                                                                      \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f1() OP z.f2(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f2() OP z.f3(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f3()                  );           \
          }                                                                      \
      }                                                                          \
    else if constexpr (LowerCut{} == Other_Upper{})                              \
      {                                                                          \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f1() OP z.f2(),                              \
                            LowerCut{},             /*Other_Upper{}*/            \
                                       w.f2() OP z.f3(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f3()                  );           \
      }                                                                          \
    else                                                                         \
      {                                                                          \
            return branched(Var<dim>{},                                          \
                                       w.f1() OP z.f1(),                              \
                                                    Other_Lower{},               \
                                       w.f1() OP z.f2(),                              \
                                                    Other_Upper{},               \
                                       w.f1() OP z.f3(),                              \
                            LowerCut{},                                          \
                                       w.f2() OP z.f3(),                              \
                            UpperCut{},                                          \
                                       w.f3() OP z.f3()                  );           \
      }                                                                          \
  }                                                                              \


  SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(+);
  SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(-);
  SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(*);
  SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(/);
  SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS(^);

  #undef SIMBPOLIC_EXACT_INTERVAL_SELF_OPERATORS  

  #define SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(OP)                          \
  template <class A, class B, class C, indexer dim, class LowerCut, class UpperCut, class F1, class F2, class OtherCut, \
            typename std::enable_if_t<is_exact<OtherCut> && is_exact<LowerCut> && is_exact<UpperCut>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const interval_function<A, B, C, dim, LowerCut, UpperCut> &w, \
                                                            const branch_function<F1, F2, dim, OtherCut> &z) \
  {                                                                              \
    if constexpr (OtherCut{} < LowerCut{})                                       \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), OtherCut{}, w.f1() OP z.f2(), LowerCut{}, w.f2() OP z.f2(), UpperCut{}, w.f3() OP z.f2() ); \
      }                                                                          \
    else if constexpr (OtherCut{} == LowerCut{})                                 \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f2() OP z.f2(), UpperCut{}, w.f3() OP z.f2() ); \
      }                                                                          \
    else if constexpr (OtherCut{} < UpperCut{})                                  \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f2() OP z.f1(), OtherCut{}, w.f2() OP z.f2(), UpperCut{}, w.f3() OP z.f2() ); \
      }                                                                          \
    else if constexpr (OtherCut{} == UpperCut{})                                 \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f2() OP z.f1(), UpperCut{}, w.f3() OP z.f2() ); \
      }                                                                          \
    else                                                                         \
      {                                                                          \
        return branched(Var<dim>{}, w.f1() OP z.f1(), LowerCut{}, w.f2() OP z.f1(), UpperCut{}, w.f3() OP z.f1(), OtherCut{}, w.f3() OP z.f2() ); \
      }                                                                          \
  }                                                                              \


  SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(+);
  SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(-);
  SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(*);
  SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(/);
  SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS(^);

  #undef SIMBPOLIC_EXACT_INTERVAL_BRANCH_OPERATORS  


  #define SIMBPOLIC_INTERVAL_OTHER_OPERATORS(OP)        \
  template<class F, class A, class B, class C, indexer dim, class LowerCut, class UpperCut, \
           typename std::enable_if_t<!is_branched<F> && !is_exceptional<F>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const F& f, const interval_function<A, B, C, dim, LowerCut, UpperCut>& intv) \
  {                                                           \
    const auto g_1 = f OP intv.f1();                          \
    const auto g_2 = f OP intv.f2();                          \
    const auto g_3 = f OP intv.f3();                          \
    return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, intv.lower_cut(), intv.upper_cut()}; \
  }                                                           \
  template<class F, class A, class B, class C, indexer dim, class LowerCut, class UpperCut, \
           typename std::enable_if_t<!is_branched<F> && !is_exceptional<F>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const interval_function<A, B, C, dim, LowerCut, UpperCut>& intv, const F& f) \
  {                                                           \
    const auto g_1 = intv.f1() OP f;                            \
    const auto g_2 = intv.f2() OP f;                            \
    const auto g_3 = intv.f3() OP f;                            \
    return interval_function<decltype(g_1), decltype(g_2), decltype(g_3), dim, LowerCut, UpperCut>{g_1, g_2, g_3, intv.lower_cut(), intv.upper_cut()}; \
  }                                                           \

  SIMBPOLIC_INTERVAL_OTHER_OPERATORS(+);
  SIMBPOLIC_INTERVAL_OTHER_OPERATORS(-);
  SIMBPOLIC_INTERVAL_OTHER_OPERATORS(*);
  SIMBPOLIC_INTERVAL_OTHER_OPERATORS(/);
  SIMBPOLIC_INTERVAL_OTHER_OPERATORS(^);

  #undef SIMBPOLIC_INTERVAL_OTHER_OPERATORS

}

#endif
