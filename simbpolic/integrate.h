#ifndef SIMBPOLIC_INTEGRATE
#define SIMBPOLIC_INTEGRATE

namespace Simbpolic
{

  template <class Func, indexer dim, class StartT, class EndT>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto integrate(const Func& f, const Var<dim> &var, const StartT &start, const EndT &end)
  {
    if constexpr (Func::template has_dimension<dim>())
      {
        //const auto dist = distribute<7>(f);
        //OPTIMIZATION TO DO: Adjust this better.
        
        const auto prim = f.template primitive<dim>();
        const auto eval_end = prim.template evaluate_along_dim<dim>(end);
        const auto eval_start = prim.template evaluate_along_dim<dim>(start);
        const auto ret = eval_end - eval_start;
        return ret;
        //Evaluates along a dimension,
        //keeping all others as variables
      }
    else
      {
        const auto diff = end - start;
        const auto ret = f * diff;
        return ret;
      }
  }
  
  /*!
    \remark integrate(f, x, a(y), b(y), y, c(z), d(z)) will do \int_{c(z)}^{d(z)} \int_{a(y)}^{b(y)} f(x, y) dx dy
  */
  template <class Func, indexer dim1, class StartT1, class EndT1, class ... Others>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto integrate(const Func& f, const Var<dim1> &var, const StartT1 &start, const EndT1 &end, const Others& ... rest)
  {
    const auto integrand = integrate(f, var, start, end);
    return integrate(integrand, rest...);
  }
}

#endif
