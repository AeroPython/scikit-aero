# coding: utf-8

"""
Decorators for multiple purposes.

"""

from __future__ import absolute_import, division

from functools import update_wrapper, wraps


def implicit(f):
    """Returns implicit function from a single-variable function.

    The form of the implicit function is

    f(t) -> F(t; x) = f(t) - x

    so the equation F(t; x) = 0 gives x = f(t).

    """

    @wraps(f)
    def _F(t, y):
        return f(t) - y

    return _F
