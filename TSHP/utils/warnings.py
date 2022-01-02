global logging

from logging  import debug
from logging  import info
from logging  import warning
from logging  import error
from logging  import critical
from logging  import exception

import logging

def debugif(condition, message, *args, **kwargs):
    if condition:
        debug(message, *args, **kwargs)

def infoif(condition, message, *args, **kwargs):
    if condition:
        info(message, *args, **kwargs)

def warnif(condition, message, *args, **kwargs):
    if condition:
        warning(message, *args, **kwargs)

def errorif(condition, message, *args, **kwargs):
    if condition:
        error(message, *args, **kwargs)

def criticalif(condition, message, *args, **kwargs):
    if condition:
        critical(message, *args, **kwargs)

def exceptionif(condition, message, *args, **kwargs):
    if condition:
        exception(message, *args, **kwargs)

def setLevel(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.getLogger().setLevel(level=numeric_level)

import contextlib
@contextlib.contextmanager
def withLevel(loglevel):
    orig_level = logging.getLogger().level
    
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.getLogger().setLevel(level=numeric_level)
    yield
    logging.getLogger().setLevel(level=orig_level)

print0 = debug
print1 = info
print2 = warning
print3 = error
print4 = critical
print4exc = exception

print0if = debugif
print1if = infoif
print2if = warnif
print3if = errorif
print4if = criticalif
print4excif = exceptionif
