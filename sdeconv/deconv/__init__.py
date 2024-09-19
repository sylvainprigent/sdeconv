"""Module that implements the image deconvolution algorithms"""
from .interface import SDeconvFilter
from .wiener import SWiener, swiener
from .richardson_lucy import SRichardsonLucy, srichardsonlucy
from .spitfire import Spitfire, spitfire
from .noise2void import Noise2VoidDeconv
from .self_supervised_nn import SelfSupervisedNNDeconv
from .nn_deconv import NNDeconv

__all__ = ['SDeconvFilter',
           'SWiener',
           'swiener',
           'SRichardsonLucy',
           'srichardsonlucy',
           'Spitfire',
           'spitfire',
           'Noise2VoidDeconv',
           'SelfSupervisedNNDeconv',
           'NNDeconv'
         ]
