# Introduction
A simple, header-only library for compile-time symbolic computation, written in C++17.

Currently only supports integration and derivation of polynomial or piecewise polynomial functions, in any number of dimensions.

# Requirements
Simbpolic requires a C++17-compatible compiler and the corresponding standard library. No further dependencies.

The code has been written with CUDA compability in mind, although it is largely untested as of the moment of the writing. In any case, if trying to use Simbpolic with CUDA, please ensure the C++17 standard is available (e. g. use at least CUDA 11 or Clang).

The users are kindly encouraged to report any problems thay may arise to the author by submitting an issue to this repository.

# Using Simbpolic
To use Simbpolic, simply `#include "simbpolic.h"` and use the relevant templates:

* `Simbpolic::Constant{val}`: A constant with runtime specified value `val`.
* `Simbpolic::Zero{}`: The neutral element of addition and subtraction and the absorbing element of multiplication.
* `Simbpolic::One{}`: The neutral element of multiplication.
* `Simbpolic::Rational<num, denom>{}`: The exact value of `num / denom`
* `Simbpolic::Intg<i>{}`: The exact integer `i` (technically a typedef to `Simbpolic::Rational<i, 1>`)
* `Simbpolic::Monomial<p, dim>{}`: The variable with index `dim` to the `p`-th power; for example, `y^3` should be written as `Simbpolic::Monomial<3, 2>{}`
* `Simbpolic::Var<dim>`: Specifies the variable with index `dim`. Not actually a function, but useful to express variable changes, integrations and so on (see below).

All of these classes, except `Simbpolic::Var` support all usual algebric operators, together with an overloaded `^` to signify exponentiation (though operator precedence rules imply one should write, e. g.,  `(x^3) * 6` to avoid getting `x^18`...).

There are some functions that allow for more elaborate functions, namely:

* ...

By default, Simbpolic uses `double` as the variable type (`ResultType`) and `int` as the variable for holding integer values (`IntegerType`). However, should the user need to change these definitions, a very convenient way of achiving this is provided: by defining a class `Simbpolic::Configuration` before `#include "simbpolic.h"` with a public `ResultType` or `IntegerType` typedef, the specified types will be used instead of the defaults. If the types are ommitted, the default values will be used.

For example:

```C++
namespace Simbpolic
{
  class Configuration
  {
    public:
      using ResultType = SomeFloatingType;
      using IntegerType = SomeIntegerType;
  };
}
```



Obviously, for any of this to work, the `simbpolic.h` file and the `simbpolic` folder must be placed in a location where the compiler or build system knows where to look for header files, but, given the diversity of choices in that area, the author will relay the responsibility of ensuring that to the user (or whomever set up the build enviroment the user is working in).

# Warnings and Caveats
Since all the functions and operations are specified using template metaprogramming, the usage of the `auto` keyword is more or less essential.

To reduce memory consumption, whenever possible, use `Simbpolic::Rational` (or `Simbpolic::Intg`) instead of floating point numbers. Otherwise, on sufficiently complicated functions (say, a three-dimensional piecewise function with tens of pieces, all with numerical coefficients), it is quite possible to get stack overflows. 

The use of especially complicated functions may lead to a very significant increase in compilation times.

Once again, the author kindly requests that any possible problems that may arise be reported by submitting an issue to this repository.

# Acknowledgements
The author must thank all those family members who kindly did not complain too much about the strange sleeping schedules during the development of this library (concurrent with other work...).

# Future Work
Include more complete Doxygen annotations and generate a more comprehensive documentation.

Validate more thoroughly the mathematical results. (User input welcome!)

Investigate possible compilation time optimizations (expression simplification, constant propagation).

Expand the symbolic capabilities (more functions, conditions in the piecewise functions, series, and so on).

Accomodate non-scalar functions (e. g. vectorial operations).

# Author Contacts
The author can be reached through the e-mail provided on the GitHub profile.
