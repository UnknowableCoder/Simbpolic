#ifndef SIMBPOLIC
#define SIMBPOLIC

/*******************************
 *          SIMBPOLIC          *
 *  Simple Symbolic Calculus   *
 *       at compile-time       *
 *                             *
 * (Currently WIP, largely     *
 *  incomplete, messy and      *
 *  underdocumented, but       *
 *  entirely functional for    *
 *  the purposes of the        *
 *  current project... )       *
 *                             *
 *  By: Nuno Fernandes         *
 *                             *
 *******************************/

#include <limits>
#include <utility>
#include <type_traits>
#include <numeric>


#if __CUDA_ARCH__
#define SIMBPOLIC_CUDA_AVAILABLE 1
#elif __CUDA__
#define SIMBPOLIC_CUDA_AVAILABLE 1
#else
#define SIMBPOLIC_CUDA_AVAILABLE 0
#endif

//For maximum interoperatability between CUDA and normal code.
#if SIMBPOLIC_CUDA_AVAILABLE
#define SIMBPOLIC_CUDA_HOS_DEV __host__ __device__
#else
#define SIMBPOLIC_CUDA_HOS_DEV
#endif

#if SIMBPOLIC_CUDA_AVAILABLE
#define SIMBPOLIC_CUDA_ONLY_HOS __host__
#else
#define SIMBPOLIC_CUDA_ONLY_HOS
#endif

#if SIMBPOLIC_CUDA_AVAILABLE
#define SIMBPOLIC_CUDA_ONLY_DEV __device__
#else
#define SIMBPOLIC_CUDA_ONLY_DEV
#endif

namespace Simbpolic
{  
  
  /*!
    \brief Allows customization of the library's behaviour by declaring member typedefs/type aliases.
    
    \detail Supported aliases:
              - \c ResultType, the type of the numeric results.
                Should be able to hold real results (so, floating point).
              - \c IntegerType, the type that holds integer quantities.
                Must be signed!
              
    \remark The user can specify only some of the aliases, with the default
            values being taken if the alias is absent.
            The full default configuration would be equivalent to using:
~~~~~~{cpp}
class Simbpolic::Configuration
{
  public:
    using ResultType = double;
    using IntegerType = int;
};
~~~~~~
    \remark Obviously, this will be equivalent to `class Configuration {};`,
            since the default values are taken when the members are not aliased.
  */
  class Configuration;
  
  namespace internals
  {
  
#define TYPEDEF_CHECKER(MEMBERNAME, INVALID_TYPE)                            \
template<class T> inline static constexpr                                    \
auto MEMBERNAME ## _t_checker(T*) -> decltype(typename T:: MEMBERNAME{});    \
template<class T> inline static constexpr                                    \
INVALID_TYPE MEMBERNAME ## _t_checker(...);                                  \
template <class T>                                                           \
using MEMBERNAME ## _t_type = decltype( MEMBERNAME ## _t_checker<T>(nullptr) );  \
template <class T> inline static constexpr                                   \
bool MEMBERNAME ## _t_exists = !std::is_same_v<MEMBERNAME ## _t_type<T>, INVALID_TYPE>; \

    class config_handler
    {
      private:
      
      TYPEDEF_CHECKER(ResultType, double);
      TYPEDEF_CHECKER(IntegerType, int);
      
      public:
      
      using Type = ResultType_t_type<Configuration>;
      using indexer = IntegerType_t_type<Configuration>;
    };

#undef TYPEDEF_CHECKER

  }
  
  
  using Type = internals::config_handler::Type;
  using indexer = internals::config_handler::indexer;
  
  namespace internals
  {
    template <class base_T, class exp_T>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline static base_T fastpow_in(const base_T &base, exp_T exp)
    {
      base_T r{1}, v{base};
      while (exp)
        {
          if (exp & 1)
            {
              r = r*v;
            }
          v = (v*v);
          exp = (exp >> 1);
        }
      return r;
    }
  }
  
  
  /*!
    \brief Expectably efficient implementation of exponentiation by integers.
    
    \pre \p exp_T must be an integer type with bitwise operations.
  */
  
  template <class base_T, class exp_T>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static base_T fastpow(const base_T &base, exp_T exp)
  {
    if (exp < 0)
      {
        return base_T(1)/internals::fastpow_in(base, -exp);
      }
    else
      {
        return internals::fastpow_in(base, exp);
      }
  }
  /*
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static indexer gcd(const indexer &a, const indexer &b)
  //The Binary Euclidean Algorithm.
  //May or may not have been implemented
  //with direct inspiration from Wikipedia...
  {
    if (a < 0 && b < 0)
      {
        return gcd(-a, -b);
      }
    else if (a < 0)
      {
        return gcd(-a, b);
      }
    else if (b < 0)
      {
        return gcd(a, -b);
      }
    if (a == b)
      {
        return a;
      }
    else if (a == 0)
      {
        return b;
      }
    else if (b == 0)
      {
        return a;
      }
    indexer u(a), v(b), factor(1);
    while (u % 2 == 0 && v % 2 == 0)
      {
        u /= 2;
        v /= 2;
        factor *= 2;
      }
    while (1)
    //Will terminate eventually.
      {
        if (u > v)
          {
            const indexer temp = u;
            u = v;
            v = temp;
          }
        v -= u;
        if (v == 0)
          {
            return u * factor;
          }
        else
          {
            while (v % 2 == 0)
              {
                v /= 2;
              }
          }
      }
    return 0;
  }
  */
  
  /*!
    \brief Base class, to mark every symbolic function implemented here.
  */
  struct SymBase
  {
  };
  
  /*!
    \brief Base class for things that represent exact numbers
    (like rationals and the currently unsupported radicals or mathematical constants)
  */
  struct SymExactNumber
  {
  };
  
  struct SymBranched
  {
  };
  
  struct SymExactBranched: public SymBranched
  {
  };
  
  struct SymHoldsValues
  {
  };
  
  struct SymOpFunc
  {
  };
  
  
  template <class T>
  inline static constexpr bool holds_values = std::is_base_of_v<SymHoldsValues, std::decay_t<T>>;

  template <class A, class B> struct func_add;
  template <class A, class B> struct func_sub;
  template <class A, class B> struct func_mul;
  template <class A, class B> struct func_div;
  
  
  struct Constant;
  struct Zero;
  struct One;
  template <indexer num_, indexer denom_> struct Rational;
  template <indexer val> using Intg = Rational<val, 1>;
  
  template <indexer store_idx> struct Stored;
  

  template <indexer order, indexer dim = 0> struct Monomial;

  template <class A, class B, indexer dim, class Cut> struct branch_function;
  template <class A, class B, class C, indexer dim, class LowerCut, class UpperCut> struct interval_function;

  template <class T>
  inline static constexpr bool is_symbolic = std::is_base_of_v<SymBase, std::decay_t<T>>;
  
  template <class T>
  inline static constexpr bool is_numeric = std::is_base_of_v<SymExactNumber, std::decay_t<T>> ||
                                            std::is_convertible_v<std::decay_t<T>, Type> ||
                                            std::is_same_v<std::decay_t<T>, Constant>;
                                            
  
  template <class A, class B>
  inline static constexpr bool is_numeric<func_add<A, B>> = is_numeric<A> && is_numeric<B>;
  
  template <class A, class B>
  inline static constexpr bool is_numeric<func_sub<A, B>> = is_numeric<A> && is_numeric<B>;
  
  template <class A, class B>
  inline static constexpr bool is_numeric<func_mul<A, B>> = is_numeric<A> && is_numeric<B>;
  
  template <class A, class B>
  inline static constexpr bool is_numeric<func_div<A, B>> = is_numeric<A> && is_numeric<B>;
  
  template <class A>
  inline static constexpr bool is_stored = false;
  
  template <indexer idx>
  inline static constexpr bool is_stored<Stored<idx>> = true;
    
  template <class T>
  inline static constexpr bool is_exact = std::is_base_of_v<SymExactNumber, std::decay_t<T>>;
  
  template <class T>
  inline static constexpr bool is_exceptional = std::is_same_v<T, Zero> || std::is_same_v<T, One>;
    
  template <class T>
  inline static constexpr bool is_branched = std::is_base_of_v<SymBranched, std::decay_t<T>>;
  
  template <class T>
  inline static constexpr bool is_exact_branched = std::is_base_of_v<SymExactBranched, std::decay_t<T>>;
  
  template <class T>
  inline static constexpr bool is_op_func = std::is_base_of_v<SymOpFunc, std::decay_t<T>>;
  
  /*!
    \brief A variable along a dimension.
    
    \remark Dimensions are 1-indexed!
  */
  template <indexer dim>
  struct Var
  {
    static_assert(dim > 0, "Dimensions are 1-indexed!");
    SIMBPOLIC_CUDA_HOS_DEV constexpr Var()
    {
    }
    
  
    SIMBPOLIC_CUDA_HOS_DEV constexpr Var(const Monomial<1, dim> &m)
    {
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline operator Monomial<1, dim> () const
    {
      return Monomial<1, dim>{};
    }
  };
  
  namespace internals
  {
    template <class Func1, class Func2, indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool share_dimensions_impl()
    {
      if constexpr (Func1::template has_dimension<dim>() && Func2::template has_dimension<dim>())
        {
          return true;
        }
      else
        {
          if constexpr (dim > Func1::max_dimension || dim > Func2::max_dimension)
            {
              return false;
            }
          else
            {
              return share_dimensions_impl<Func1, Func2, dim + 1>();
            }
        }
    }
  }
  
  /*!
    \brief Returns `true` if the functions share at least one dimension.
  */
  template <class Func1, class Func2>
  SIMBPOLIC_CUDA_HOS_DEV inline static constexpr bool share_dimensions()
  {
    if constexpr (Func1::min_dimension > Func2::max_dimension || 
                  Func1::max_dimension < Func2::max_dimension    )
      {
        return false;
      }
    else
      {
        if constexpr (Func1::min_dimension < Func2::min_dimension)
          {
            return internals::share_dimensions_impl<Func1, Func2, Func2::min_dimension>();
          }
        else
          {
            return internals::share_dimensions_impl<Func1, Func2, Func1::min_dimension>();
          }
      }
  }
  
  /*!
    \remark To actually get stuff from the store, it must have a function with signature
            `template <indexer i> SIMBPOLIC_CUDA_HOS_DEV inline constexpr Type get() const`
            that returns the value that is to be attributed to the `i`-th constant.
  */
  struct Store {};
  
  template <class C>
  inline static constexpr bool is_store = std::is_base_of_v<Store, C>;
  
  
  namespace internals
  {
    class NotFoundType{};

#define SIMBPOLIC_HOLD_ARGS(...) __VA_ARGS__

#define SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(FUNCNAME, TEMPARGS, TEMPARGNAMES, ARGINIT) \
private:                                                                                    \
  template <class T, TEMPARGS> inline static constexpr                                      \
  auto FUNCNAME ## _checker(T*) -> decltype(std::declval<T>().template FUNCNAME<TEMPARGNAMES>(ARGINIT)); \
  template <class T, TEMPARGS> inline static constexpr                                      \
  auto FUNCNAME ## _checker(...) -> NotFoundType;                                           \
public:                                                                                     \
  template <TEMPARGS> static constexpr bool                                                 \
  has_ ##FUNCNAME = !std::is_same_v<NotFoundType, decltype(FUNCNAME ## _checker <C, TEMPARGNAMES> (nullptr))>

    template <class C>
    class MemberFunctionChecker
    {
      SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(change_dim,
                                               SIMBPOLIC_HOLD_ARGS(indexer from, indexer to),
                                               SIMBPOLIC_HOLD_ARGS(from, to),
                                               SIMBPOLIC_HOLD_ARGS(Var<from>{}, Var<to>{})
                                              );
                                              
      SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(offset,
                                               SIMBPOLIC_HOLD_ARGS(indexer dim, class Off),
                                               SIMBPOLIC_HOLD_ARGS(dim, Off),
                                               SIMBPOLIC_HOLD_ARGS(Var<dim>{}, Off{})
                                              );
                                              
      SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(reverse,
                                               SIMBPOLIC_HOLD_ARGS(indexer dim),
                                               SIMBPOLIC_HOLD_ARGS(dim),
                                               SIMBPOLIC_HOLD_ARGS(Var<dim>{})
                                              );
                                              
      SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(deform,
                                               SIMBPOLIC_HOLD_ARGS(indexer dim, class Fact),
                                               SIMBPOLIC_HOLD_ARGS(dim, Fact),
                                               SIMBPOLIC_HOLD_ARGS(Var<dim>{}, Fact{})
                                              );
                                              
      SIMBPOLIC_MEMBER_FUNCTION_CHECKER_HELPER(distribute,
                                               SIMBPOLIC_HOLD_ARGS(indexer recurse_count),
                                               SIMBPOLIC_HOLD_ARGS(recurse_count),
                                               
                                              );
    };
  }

  /*!
    \brief Changes any functions that depend on the \p from variable to \p to.
  */
  template <indexer from, indexer to, class Func>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto change_dim(const Var<from> &x, const Var<to> &y, const Func &f)
  {
    if constexpr (internals::MemberFunctionChecker<Func>::template has_change_dim<from, to>)
      {
        return f.change_dim(x, y);
      }
    else
      {
        return f;
      }
  }

  /*!
    \brief Returns `f(x + off)` in a consistent way.
           (For branched functions mainly, since we don't properly handle conditions yet...)
  */
  template <indexer dim, class Func, class Off>
  SIMBPOLIC_CUDA_HOS_DEV constexpr static auto offset(const Var<dim> &var, const Off &off, const Func& f)
  {
    if constexpr (internals::MemberFunctionChecker<Func>::template has_offset<dim, Off>)
      {
        return f.offset(var, off);
      }
    else
      {
        return f;
      }
  }
  
  /*!
    \brief Returns `f(-x)` in a consistent way.
           (For branched functions mainly, since we don't properly handle conditions yet...)
  */
  template <indexer dim, class Func>
  SIMBPOLIC_CUDA_HOS_DEV constexpr static auto reverse(const Var<dim> &var, const Func& f)
  {
    if constexpr (internals::MemberFunctionChecker<Func>::template has_reverse<dim>)
      {
        return f.reverse(var);
      }
    else
      {
        return f;
      }
  }
  
  /*!
    \brief Returns `f(fact x)` in a consistent way.
           (For branched functions mainly, since we don't properly handle conditions yet...)
           
    \warning The factor must be positive! If not, the conditions will not be properly updated!
             In case a negative factor is desired, see `reverse`.
  */
  template <indexer dim, class Func, class Fact>
  SIMBPOLIC_CUDA_HOS_DEV constexpr static auto deform(const Var<dim> &var, const Fact &fact, const Func& f)
  {
    if constexpr (internals::MemberFunctionChecker<Func>::template has_deform<dim, Fact>)
      {
        return f.deform(var, fact);
      }
    else
      {
        return f;
      }
  }
  
  template <indexer recurse_count, class Func>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline static auto distribute(const Func& f)
  {
    if constexpr (recurse_count > 0 && internals::MemberFunctionChecker<Func>::template has_distribute<recurse_count>)
      {
        return f.template distribute<recurse_count>();
      }
    else
      {
        return f;
      }
  }
}

#include "simbpolic/constants.h"
#include "simbpolic/constant_comparison.h"
#include "simbpolic/stored.h"
#include "simbpolic/func_holders.h"
#include "simbpolic/monomial.h"
#include "simbpolic/op_funcs.h"
#include "simbpolic/branch.h"
#include "simbpolic/interval.h"
#include "simbpolic/branching_helper.h"
#include "simbpolic/integrate.h"

namespace Simbpolic
{

#define SIMBPOLIC_OPS_BINARY_OPER(OP, NAME)                                           \
template <class T1, class T2,                                                         \
          typename std::enable_if_t<(is_symbolic<T1> || is_numeric<T1>) &&            \
                                    (is_symbolic<T2> || is_numeric<T2>) &&            \
                                    !(is_branched<T1> && !is_branched<T2>) &&         \
                                    !(!is_branched<T1> && is_branched<T2>)>* = nullptr> \
SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const T1& a, const T2& b)   \
{                                                                                     \
  if constexpr (is_symbolic<T1>&& is_symbolic<T2>)                                    \
    {                                                                                 \
      return NAME <std::decay_t<T1>,std::decay_t<T2>>{a, b};                          \
    }                                                                                 \
  else if constexpr (is_symbolic<T1>)                                                 \
    {                                                                                 \
      return NAME <std::decay_t<T1>,Constant>{a, Constant{Type(b)}};                  \
    }                                                                                 \
  else /*if constexpr (is_symbolic<T2>)*/                                             \
    {                                                                                 \
      return NAME <Constant,std::decay_t<T2>>{Constant{Type(b)},a};                   \
    }                                                                                 \
}                                                                                     \
  
  SIMBPOLIC_OPS_BINARY_OPER(+, func_add);
  SIMBPOLIC_OPS_BINARY_OPER(-, func_sub);
  SIMBPOLIC_OPS_BINARY_OPER(*, func_mul);
  SIMBPOLIC_OPS_BINARY_OPER(/, func_div);
    
  template <class T, typename std::enable_if_t<is_symbolic<T>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const T& t)
  {
    return t;
  }
  
  template <class T, typename std::enable_if_t<is_symbolic<T>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const T& t)
  {
    return t * Intg<-1>{};
  }
  
  
#define SIMBPOLIC_OPS_BINARY_OPER_SIMPLIFIER(OP, NAME)                                \
template <class T1, class A, class B,                                                 \
          typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> && \
                                     (is_numeric<A> || is_numeric<B>) >* = nullptr>   \
SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const T1& val, const NAME<A, B>& func) \
{                                                                                     \
  if constexpr (is_numeric<A> && !is_stored<A>)                                       \
    {                                                                                 \
      return (val OP func.f1()) OP func.f2();                                         \
    }                                                                                 \
  else /*if constexpr (is_numeric<B> && !is_stored<B>) */                             \
    {                                                                                 \
      return func.f1() OP (val OP func.f2());                                         \
    }                                                                                 \
}                                                                                     \
template <class T1, class A, class B,                                                 \
          typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1>  && !is_op_func<T1> && \
                                     (is_numeric<A> || is_numeric<B>) >* = nullptr>   \
SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const NAME<A, B>& func, const T1& val) \
{                                                                                     \
  if constexpr (is_numeric<A> && !is_stored<A>)                                       \
    {                                                                                 \
      return (func.f1() OP val) OP func.f2();                                         \
    }                                                                                 \
  else /*if constexpr (is_numeric<B> && !is_stored<B>) */                             \
    {                                                                                 \
      return func.f1() OP (func.f2() OP val);                                         \
    }                                                                                 \
}                                                                                     \
  
  SIMBPOLIC_OPS_BINARY_OPER_SIMPLIFIER(+, func_add);
  SIMBPOLIC_OPS_BINARY_OPER_SIMPLIFIER(*, func_mul);
    
  template <class A, class B,
            typename std::enable_if_t< is_numeric<A> || is_numeric<B> >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator + (const One& val, const func_add<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val + func.f1()) + func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val + func.f2()) + func.f1() ;
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const T1& val, const func_add<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val - func.f1()) - func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val - func.f2()) - func.f1() ;
      }
  }

  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator + (const T1& val, const func_sub<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val + func.f1()) - func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val - func.f2()) + func.f1() ;
      }
  }

  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const T1& val, const func_sub<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val - func.f1()) + func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val + func.f2()) - func.f1() ;
      }
  } 
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const T1& val, const func_mul<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val / func.f1()) / func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val / func.f2()) / func.f1() ;
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const T1& val, const func_div<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val / func.f1()) * func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val * func.f2()) / func.f1() ;
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator * (const T1& val, const func_div<A, B>& func)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (val * func.f1()) / func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (val / func.f2()) * func.f1() ;
      }
  }
  
  template <class A, class B,
            typename std::enable_if_t< is_numeric<A> || is_numeric<B> >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator + (const func_add<A, B>& func, const One& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() + val) + func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return (func.f2() + val) + func.f1() ;
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const func_add<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() - val) + func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() + (func.f2() - val);
      }
  }

  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator + (const func_sub<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() + val) - func.f2();
      }
      
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() - (func.f2() - val);
      }
  }

  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !std::is_same_v<T1, Zero> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const func_sub<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() - val) - func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() - (func.f2() + val);
      }
  } 
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const func_mul<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() / val) * func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() * (func.f2() / val) ;
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const func_div<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1()/val) / func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() / (func.f2() * val);
      }
  }
  
  template <indexer n, indexer d, class A, class B,
            typename std::enable_if_t< is_numeric<A> || is_numeric<B> >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const func_mul<A, B>& func, const Rational<n,d>& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() / val) * func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() * (func.f2() / val) ;
      }
  }
  
  template <indexer n, indexer d, class A, class B,
            typename std::enable_if_t< is_numeric<A> || is_numeric<B> >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const func_div<A, B>& func, const Rational<n,d>& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1()/val) / func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() / (func.f2() * val);
      }
  }
  
  template <class T1, class A, class B,
            typename std::enable_if_t<is_numeric<T1> && !is_exceptional<T1> && !is_op_func<T1> &&
                                      (is_numeric<A> || is_numeric<B>) >* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator * (const func_div<A, B>& func, const T1& val)
  {
    if constexpr (is_numeric<A> && !is_stored<A>)
      {
        return (func.f1() * val) / func.f2();
      }
    else /*if constexpr (is_numeric<B> && !is_stored<B>) */
      {
        return func.f1() * (val / func.f2());
      }
  }

  template <class T, typename std::enable_if_t<is_symbolic<T> && !holds_values<T>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const T& t1, const T& t2)
  {
    return One{};
  }

  template <class T, typename std::enable_if_t<is_symbolic<T> && !holds_values<T>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator - (const T& t1, const T& t2)
  {
    return Zero{};
  }
  
  template <class T, indexer val, typename std::enable_if_t<!is_exceptional<T>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator ^ (const T& base, const Intg<val>& exp)
  {
    if constexpr (val < 0)
      {
        return Intg<1>{} / (base ^ Intg<-val>{});
      }
    else if constexpr (val == 0)
      {
        return One{};
      }
    else if constexpr (val == 1)
      {
        return base;
      }
    else
      {
        if constexpr (val % 2 == 0)
          {
            const auto temp = base ^ Intg<val/2>{};
            return temp * temp;
          }
        else
          {
            const auto temp = base ^ Intg<val - 1>{};
            return base * temp;
          }
      }
  }
  //NOTE: operator ^ has lower precedence than the arithmetic operators.
  //If using it to signify exponentiation,
  //wrap the thing in parenthesis.
  //(But it's still shorter than using pow<num>(thing)...)

  
  //And now some hacks follow for us to be able to complete the project in time.
  //A more complete treatment of piecewise functions would make them unnecessary,
  //but that would mean we need to take into account integration constants
  //and solve for the equality of the integrated function at the (possible) discontinuities.
  //Worse, we'd need to support essentially arbitrary conditions (or, at the very least, linear ones),
  //which would be much more complicated...

  
}

//#undef SIMBPOLIC_CUDA_AVAILABLE
//#undef SIMBPOLIC_CUDA_HOS_DEV
//#undef SIMBPOLIC_CUDA_ONLY_DEV
//#undef SIMBPOLIC_CUDA_ONLY_HOS
#undef SIMBPOLIC_OPS_BINARY_OPER
#undef SIMBPOLIC_OPS_BINARY_OPER_SIMPLIFIER
#undef SIMBPOLIC_OPS_OPERATORS

#endif
