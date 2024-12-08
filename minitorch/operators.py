"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiply x and y."""
    return x * y


def id(x: float) -> float:
    """Return x."""
    return x


def add(x: float, y: float) -> float:
    """Add x to y."""
    return x + y


def neg(x: float) -> float:
    """Negate x."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0
    # return x < y


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return 1.0 if x == y else 0.0
    # return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the relu of x."""
    return x if x >= 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the log of x."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exp of x."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log(x) with respect to x."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of x."""
    result = 1.0 / x
    return result


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of 1/x with respect to x."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of relu(x) with respect to x."""
    return d if x > 0 else 0


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        f: Function from one value to one value

    Returns:
    -------
        A function that takes a list, applies `f` to each element, and returns a new list


    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        f: combine two values

    Returns:
    -------
        Function that takes two equally sizes lists `ls1` and `ls2`, produce a new list by
        applying f(x,y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    return zipWith(add)(ls1, ls2)


def reduce(
    f: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        f: combine two values
        init: start value $x_0$

    Returns:
    -------
        Function that takes a list of `ls` elements
        $x_0 \ldots x_n$ and computes the reduction :math `f(x_3, f(x_2,f(x_1,x_0)))`

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = init
        for l in ls:
            val = f(val, l)
        return val

    return _reduce


# Implement the following core functions
# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg)(ls)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Take the product of a list."""
    return reduce(mul, 1.0)(xs)


# ... existing code ...
