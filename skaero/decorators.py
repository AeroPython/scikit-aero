# coding: utf-8

"""
Decorators for multiple purposes.

Copyright notice of method_decorator from Django:

Copyright (c) Django Software Foundation and individual contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    
    3. Neither the name of Django nor the names of its contributors may be used
    to endorse or promote products derived from this software without
    specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    This decorator doesn't preserve the function signature, and that is
    bad. Maybe I can use https://pypi.python.org/pypi/decorator. 
    """
    @wraps(f)
    def wrapper(self, x, *args, **kwargs):
        x_ = np.asanyarray(x)
        return f(self, x_, *args, **kwargs)
    return wrapper


def implicit(f):
    """Returns implicit function from a single-variable function.

    The form of the implicit function is

    f(t; *) -> F(t; x; *) = f(t; *) - x

    so the equation F(t; x; *) = 0 gives x = f(t; *) where * are extra
    arguments.

    """
    @wraps(f)
    def _F(t, y, *args, **kwargs):
        return f(t, *args, **kwargs) - y
    return _F


def method_decorator(decorator):
    """
    Converts a function decorator into a method decorator
    """
    # 'func' is a function at the time it is passed to _dec, but will eventually
    # be a method of the class it is defined it.
    def _dec(func):
        def _wrapper(self, *args, **kwargs):
            @decorator
            def bound_func(*args2, **kwargs2):
                return func(self, *args2, **kwargs2)
            # bound_func has the signature that 'decorator' expects i.e.  no
            # 'self' argument, but it is a closure over self so it can call
            # 'func' correctly.
            return bound_func(*args, **kwargs)
        # In case 'decorator' adds attributes to the function it decorates, we
        # want to copy those. We don't have access to bound_func in this scope,
        # but we can cheat by using it on a dummy function.
        @decorator
        def dummy(*args, **kwargs):
            pass
        update_wrapper(_wrapper, dummy)
        # Need to preserve any existing attributes of 'func', including the name.
        update_wrapper(_wrapper, func)
        
        return _wrapper
    update_wrapper(_dec, decorator)
    # Change the name to aid debugging.
    _dec.__name__ = 'method_decorator(%s)' % decorator.__name__
    return _dec
