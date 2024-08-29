"""Module that implements the image deconvolution algorithms"""
from .interface import SDeconvFilter
from .wiener import SWiener
from .richardson_lucy import SRichardsonLucy
from .spitfire import Spitfire

__all__ = ['SDeconvFilter', 
           'SWiener',
           'SRichardsonLucy',
           'Spitfire']
