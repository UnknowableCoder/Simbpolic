#ifndef SIMBPOLIC_CONSTANT_COMPARISON
#define SIMBPOLIC_CONSTANT_COMPARISON

namespace Simbpolic
{

  template <class T1, class T2, typename std::enable_if_t<is_exact<T1> && is_exact<T2>>* = nullptr >
  SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer compare(const T1& first, const T2& second)
  //If the availability of the C++20 standard could be guaranteed,
  //this would simply be the <=> operator:
  //Returns 1 if first > second, -1 if first < second, 0 otherwise.
  //
  //Should produce constexpr results as long as the methods are exact.
  //
  //And we compare through the Type so we can duly account for precision
  //in non-arithmetic cases like pi or e?
  {
    const Type a(first);
    const Type b(second);
    return (a > b) - (b > a);
  }

  template <indexer a, indexer b, indexer c, indexer d>
  SIMBPOLIC_CUDA_HOS_DEV inline static constexpr indexer compare(const Rational<a, b>& r1, const Rational<c, d>& r2)
  {
    return (a * d > b * c) - (a * d < b * c);
    //We can do this without worrying about the signs because the denominators are defined positive.
  }



  #define SIMBPOLIC_COMPARISON_OPERATORS(COMP)                           \
  template <class T1, class T2, typename std::enable_if_t<is_exact<T1> && is_exact<T2>>* = nullptr > \
  SIMBPOLIC_CUDA_HOS_DEV inline constexpr bool operator COMP (const T1& a, const T2& b) \
  {                                                                      \
    return compare(a, b) COMP 0;                                         \
  }                                                                      \

  SIMBPOLIC_COMPARISON_OPERATORS(<);
  SIMBPOLIC_COMPARISON_OPERATORS(>);
  SIMBPOLIC_COMPARISON_OPERATORS(<=);
  SIMBPOLIC_COMPARISON_OPERATORS(>=);
  SIMBPOLIC_COMPARISON_OPERATORS(==);
  SIMBPOLIC_COMPARISON_OPERATORS(!=);
}
#endif
