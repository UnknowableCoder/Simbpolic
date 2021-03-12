# Introduction
A simple, header-only library for compile-time symbolic computation, written in C++17.

Currently only supports integration and derivation of polynomial or piecewise polynomial functions (whose conditions are given by simple inequalities), in any number of dimensions.

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

* `Simbpolic::change_dim(Var<from>, Var<to>, function)`: Changes any terms of `function` that depend on the variable with index `from` to depend on the variable with index `to`
* `Simbpolic::offset(Var<dim>, offset, function)`: Changes `function(..., x_dim, ...)` to `function(..., x_dim + offset, ...)` in a manner consistent with piecewise functions
* `Simbpolic::reverse(Var<dim>, function)`: Changes `function(..., x_dim, ...)` to `function(..., -x_dim, ...)` in a manner consistent with piecewise functions
* `Simbpolic::expand(Var<dim>, factor, function)`: Changes `function(..., x_dim, ...)` to `function(..., x_dim * factor, ...)`, for factor > 0, in a manner consistent with piecewise functions
* `Simbpolic::integrate(function, Var<dim1>, a_1, b_1, ...)`: Gives the integral of `function` along `dim1` from `a_1` to `b_1`. If any additional arguments are given, integrates along other dimensions as well.
* `Simbpolic::branched(Var<dim>, f_1, k_1, f_2, ...)`: Gives the piecewise function that is `f_1` for `x_dim < k_1` and `f_2` for `x_dim > k_1`. If any additional arguments are provided (in the form `k_i, f_i`), keeps giving the branched function that is, in general, `f_(i-1)` for `k_(i-1) < x_dim < k_i` (where we can consider, to make this a really general expression, `k_0 = -\infty` and `k_n = +\infty`).

All of the symbolic functions provided by Simbpolic have the `derivative<dim>()` and `primitive<dim>()` member function, which give, respectively, the derivative and primitive along dimension `dim`, the `evaluate_along_dim<dim>(val)` which evaluate the function at `x_dim = val` (and the remaining coordinates unspecified), and an `operator(...)` which will evaluate the function with `x_i` given by the `i`-th argument (with the coordinates with index greater than the number of arguments remaining unspecified).


Obviously, for any of this to work, the `simbpolic.h` file and the `simbpolic` folder must be placed in a location where the compiler or build system knows where to look for header files, but, given the diversity of choices in that area, the author will relay the responsibility of ensuring that to the user (or whomever set up the build enviroment the user is working in).

# Configuration

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
