#ifndef SIMBPOLIC_BRANCHING_HELPER
#define SIMBPOLIC_BRANCHING_HELPER

namespace Simbpolic
{
  
  template <indexer dim, class F1>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto branched(const Var<dim>& var, const F1& f1)
  {
    return f1;
  }

  template < indexer dim, class F1, class F2, class Cut>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto branched(const Var<dim>& var, const F1& f1, const Cut& c1, const F2 &f2)
  {
    return branch_function<F1, F2, dim, Cut>{f1, f2, c1};
  }

  template < indexer dim, class F1, class F2, class F3, class LowCut, class UpCut>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto branched(const Var<dim>& var, const F1& f1,
                                                                  const LowCut& c1, const F2 &f2,
                                                                   const UpCut& c2, const F3 &f3   )
  {
    return interval_function<F1, F2, F3, dim, LowCut, UpCut>{f1, f2, f3, c1, c2};
  }
  

  /*!
    To specify something of the form:

~~~~~~
             __
            /
           /  f_1 , x_k < c_1
          |
          |   f_2 , c_1 < x_k < c_2
          |
f(x_k) = <|   f_3 , c_2 < x_k < c_3
          |
          |   ...
          | 
           \  f_n , c_n < x_k
            \__
~~~~~~

    One should write branched(Var<k>{}, f_1, c_1, f_2, c_2, f_3, c_3, ..., c_n, f_n)
        
    (For implementation reasons, functions with a domain other than |R are not supported...)
    
    \warning The caller must ensure that the order of the cut-off points is correct!
             For implementation reasons, this cannot be ensured by the functions.

  */
  template < indexer dim, class F1, class F2, class F3, class Cut1, class Cut2, class ... Others>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto branched(const Var<dim>& var, const F1& f1,
                                                               const Cut1& c1, const F2 &f2,
                                                               const Cut2& c2, const F3 &f3, const Others& ... rest   )
  {
    static_assert((sizeof...(Others) )% 2 == 0, "The number of arguments must be odd to specify all conditions");
    
    const auto first = branched(var, f1, c1, f2, c2, Zero{});
    const auto second = branched(var, Zero{}, c2, f3, rest...);
    return func_add<decltype(first), decltype(second)>{first, second};
  }
}

#endif
