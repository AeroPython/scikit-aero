# coding: utf-8

"""
Decorators for multiple purposes.

"""

from __future__ import division, absolute_import

from functools import wraps

import numpy as np


def arrayize(f):
    """
    TODO
    ----
    * Document.
    * Test.
    * Accept parameter: argument to convert to array (inspect) OR
    * Arrayize first argument and distinguish method or function (inspect).
    """
    @wraps(f)
    def wrapper(self, x, *args, **kwargs):
        x_ = np.asanyarray(x)
        return f(self, x_, *args, **kwargs)
    return wrapper
