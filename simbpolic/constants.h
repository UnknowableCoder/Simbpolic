#ifndef SIMBPOLIC_CONSTANTS
#define SIMBPOLIC_CONSTANTS

namespace Simbpolic
{

    
  struct Constant : public SymBase, public SymHoldsValues
  {
    const Type val;
    
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
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr Constant(const Type& value): val(value)
    {
    }
    
    friend std::ostream& operator << (std::ostream &s, const Constant& z)
    {
        s << z.val;
        return s;
    }

    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return (*this) * Monomial<1, dim>{};
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const;
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return Constant{val};
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return Constant{val};
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(val);
    }
  };

  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Constant& c)
  {
    return Constant{-c.val};
  }

  #define SIMBPOLIC_CONSTANT_OPERATORS(OP)                                  \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const Constant& c1, const Constant& c2) \
  {                                                                         \
    return Constant{c1.val OP c2.val};                                      \
  }                                                                         \
  template <class T2, typename std::enable_if_t<is_numeric<T2> && !is_exceptional<T2>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const Constant& c, const T2& other) \
  {                                                                         \
    return Constant{c.val OP Type(other)};                                  \
  }                                                                         \
  template <class T2, typename std::enable_if_t<is_numeric<T2> && !is_exceptional<T2>>* = nullptr> \
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator OP (const T2& other, const Constant& c) \
  {                                                                         \
    return Constant{Type(other) OP c.val};                                  \
  }                                                                         \

  SIMBPOLIC_CONSTANT_OPERATORS(+);
  SIMBPOLIC_CONSTANT_OPERATORS(-);
  SIMBPOLIC_CONSTANT_OPERATORS(*);
  SIMBPOLIC_CONSTANT_OPERATORS(/);
  
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator / (const T2& t2, const Constant &c)
  {
    return t2 * Constant{Type(1)/c.val};
  }

  #undef SIMBPOLIC_CONSTANT_OPERATORS
  
  struct Zero :
  public SymBase, public SymExactNumber
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
   
    friend std::ostream& operator << (std::ostream &s, const Zero& z)
    {
        s << Type(0);
        return s;
    }

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
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return Zero{};
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return Zero{};
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return Zero{};
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return Zero{};
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Constant() const
    {
      return Constant{Type(0)};
    }
    
    template <indexer dummy = sizeof(char)>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Rational<0,dummy> () const
    {
      return Rational<0,dummy>{};
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(0);
    }
      
  };

  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Zero& z1, const Zero& z2)
  {
    return Zero{};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Zero& z1, const Zero& z2)
  {
    return Zero{};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Zero& z1, const Zero& z2)
  {
    return Zero{};
  }
  
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Zero& z, const Constant& other)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Zero& z, const Constant& other)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Zero& z, const Constant& other)
  {
    return z;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Zero& z, const Constant& other)
  {
    return z;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Constant& other, const Zero& z)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Constant& other, const Zero& z)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Constant& other, const Zero& z)
  {
    return z;
  }
  
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Zero& z, const T2& other)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Zero& z, const T2& other)
  {
    return -other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Zero& z, const T2& other)
  {
    return z;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Zero& z, const T2& other)
  {
    return z;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const T2& other, const Zero& z)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const T2& other, const Zero& z)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const T2& other, const Zero& z)
  {
    return z;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const T2& other, const Zero& z)
  {
    static_assert(sizeof(T2)>0,"Infinities not yet supported!");
    return Zero{};
  }

  template <indexer dim>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto Constant::derivative() const
  {
    return Zero{};
  }

  struct One :
  public SymBase, public SymExactNumber
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
    
    friend std::ostream& operator << (std::ostream &s, const One& z)
    {
        s << Type(1);
        return s;
    }

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
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      return Monomial<1, dim>{};
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return Zero{};
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return One{};
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return One{};
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Constant() const
    {
      return Constant{Type(1)};
    }
    
    template <indexer dummy = sizeof(char)>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Rational<dummy,dummy> () const
    {
      return Rational<dummy,dummy>{};
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(1);
    }
    
    
  };

  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Zero& z1, const Zero& z2)
  {
    return One{};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const One& one, const One& other)
  {
    return One{};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const One& one, const One& other)
  {
    return One{};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator^ (const One& one, const One& other)
  {
    return One{};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const One& one, const Constant& other)
  {
    return Constant{Type(1) + other.val};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Constant& other, const One& one)
  {
    return Constant{other.val + Type(1)};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const One& one, const Constant& other)
  {
    return Constant{Type(1) - other.val};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Constant& other, const One& one)
  {
    return Constant{other.val - Type(1)};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const One& one, const Constant& other)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Constant& other, const One& one)
  {
    return other;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const One& one, const Constant& other)
  {
    return Constant{Type(1)/other.val};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Constant& other, const One& one)
  {
    return other;
  }
  
  
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const One& one, const Zero& z)
  {
    return z;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Zero& z, const One& one)
  {
    return z;
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Zero& z, const One& one)
  {
    return z;
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const One& one, const Zero& other)
  {
    return One{};
  }
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Zero& other, const One& one)
  {
    return One{};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const One& one, const Zero& other)
  {
    return One{};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Zero& other, const One& one)
  {
    return -One{};
  }
  
  
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const One& one, const T2& other)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const T2& other, const One& one)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const T2& other, const One& one)
  {
    return other;
  }
  template <class T2, typename std::enable_if_t<!is_exceptional<T2>>* = nullptr>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator ^ (const One& one, const T2& exp)
  {
    return One{};
  }  
    
  template <indexer val>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator ^ (const Zero&z, const Intg<val>& exp)
  {
    if constexpr (val < 0)
      {
        static_assert(val >= 0, "Infinities not yet supported!");
      }
    else if constexpr (val == 0)
      {
        return One{};
      }
    else
      {
        return Zero{};
      }
  }
    
  template <indexer num_, indexer denom_> struct Rational :
  public SymBase,
  public SymExactNumber
  {
    
    static constexpr indexer num = num_;
    static constexpr indexer denom = denom_;
    
    static_assert(denom > 0, "Denominators should be positive!");

    SIMBPOLIC_CUDA_HOS_DEV static constexpr inline auto simplify()
    {
      if constexpr (num == 0)
        {
          return Zero{};
        }
      else if constexpr (num == 1 && denom == 1)
        {
          return One{};
        }
      else
        {
          return Rational<num/std::gcd(num, denom),denom/std::gcd(num, denom)>{};
        }
    }
    
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
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr Rational()
    {
    }
    
    friend std::ostream& operator << (std::ostream &s, const Rational& z)
    {
      if constexpr (denom == 1)
        {
          s << num;
        }
      else
        {
          if constexpr (num < 0)
            {
              s << "-\\frac{" << -num << "}{" << denom << "}";
            }
          else
            {
              s << "\\frac{" << num << "}{" << denom << "}";
            }
        }
        return s;
    }

    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto primitive() const
    {
      if constexpr (num == 0)
        {
          return Zero{};
        }
      else
        {
          return simplify() * Monomial<1, dim>{};
        }
    }
    
    template <indexer dim>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto derivative() const
    {
      return Zero{};
    }
    
    template <indexer dim, class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto evaluate_along_dim(const Args& ... args) const
    {
      return simplify();
    }
    
    template <class ... Args>
    SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator() (const Args& ... args) const
    {
      return simplify();
    }
    
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Constant() const
    {
      return Constant{Type(num)/Type(denom)};
    }
    
    template <indexer dummy1 = num, indexer dummy2 = denom, typename std::enable_if_t<dummy1 == 0 && dummy2 == 1>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator Zero () const
    {
      return Zero{};
    }
    
    template <indexer dummy1 = num, indexer dummy2 = denom, typename std::enable_if_t<dummy1 == 1 && dummy2 == 1>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr operator One () const
    {
      return One{};
    }
    
    template <class T1, typename std::enable_if_t<std::is_convertible_v<Type, T1>>* = nullptr>
    SIMBPOLIC_CUDA_HOS_DEV constexpr explicit operator T1() const
    {
      return T1(Type(num)/Type(denom));
    }
    
  };
    
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const One& one, const Rational<a, b>& r1)
  {
    return Rational<b+a, b>::simplify();
  }
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const One& one, const Rational<a, b>& r1)
  {
    return Rational<b-a, b>::simplify();
  }
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Rational<a, b>& r1, const One& one)
  {
    return Rational<a+b, b>::simplify();
  }
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Rational<a, b>& r1, const One& one)
  {
    return Rational<a-b, b>::simplify();
  }
  
  
  template <indexer num, indexer den>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Rational<num, den>& r)
  {
    return Rational<-num, den>::simplify();
  }
  template <indexer a, indexer b, indexer c, indexer d>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const Rational<a, b>& r1, const Rational<c, d>& r2)
  {
    return Rational<a*d + b*c, b*d>::simplify();
  }
  template <indexer a, indexer b, indexer c, indexer d, typename std::enable_if_t<a != c || b != d>* = nullptr>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const Rational<a, b>& r1, const Rational<c, d>& r2)
  {
    return Rational<a*d - b*c, b*d>::simplify();
  }
  template <indexer a, indexer b, indexer c, indexer d>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator* (const Rational<a, b>& r1, const Rational<c, d>& r2)
  {
    return Rational<a*c, b*d>::simplify();
  }
  
  template <class T2, indexer a, indexer b, typename std::enable_if_t<!is_exceptional<T2> && !is_op_func<T2>>* = nullptr>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const T2& other, const Rational<a, b>& r)
  {
    if constexpr (a < 0)
      {
        return other * Rational<-b,-a>::simplify();
      }
    else if constexpr (a > 0)
      {
        return other * Rational<b,a>::simplify();
      }
    else
      {
        static_assert(a != 0, "Infinities not yet supported!");
        return Zero{};
      }
  }
  
  template <indexer a, indexer b, indexer c, indexer d, typename std::enable_if_t<a != c || b != d>* = nullptr>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Rational<a, b>& r1, const Rational<c, d>& r2)
  {
    return Rational<a * d * (c < 0 ? -1 : 1), b * c * (c < 0 ? -1 : 1)>{};
  }
  
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const Rational<a, b>& r1, const Rational<a, b>& r2)
  {
    return One{};
  }
  
  template <indexer a, indexer b>  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator/ (const One& other, const Rational<a, b>& r)
  {
    if constexpr (a < 0)
      {
        return Rational<-b,-a>::simplify();
      }
    else if constexpr (a > 0)
      {
        return Rational<b,a>::simplify();
      }
    else
      {
        static_assert(a != 0, "Infinities not yet supported!");
        return Zero{};
      }
  }
  template <indexer num, indexer den, indexer val>
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator ^ (const Rational<num, den>& r, const Intg<val>& exp)
  {
    return Rational<fastpow(num, val), fastpow(den, val)>::simplify();
  }


  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator+ (const One& one, const One& other)
  {
    return Intg<2>{};
  }
  
  SIMBPOLIC_CUDA_HOS_DEV constexpr inline auto operator- (const One& one, const One& other)
  {
    return Zero{};
  }

}

#endif
