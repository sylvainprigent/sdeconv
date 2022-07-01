"""Module that implements the image deconvolution algorithms"""
from .wiener import SWiener
from .richardson_lucy import SRichardsonLucy
from .spitfire import Spitfire

__all__ = ['SWiener', 'SRichardsonLucy', 'Spitfire']
