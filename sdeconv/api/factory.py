"""Implements factory for PSF and deconvolution modules"""

import numpy as np
import torch

from ..psfs import SPSFGenerator
from ..deconv import SDeconvFilter


class SDeconvFactoryError(Exception):
    """Raised when an error happen when a module is built in the factory"""


class SDeconvModuleFactory:
    """Factory for SDeconv modules"""
    def __init__(self):
        self._data = {}

    def register(self, key: str, metadata: dict[str, any]):
        """Register a new builder to the factory

        :param key: Name of the module to register
        :param metadata: Dictionary containing the filter metadata
        """
        self._data[key] = metadata

    def get_parameters(self, key: str) -> dict[str, any]:
        """Parameters getter method
        
        :param key: Name of the module builder
        :return: The module parameters
        """
        return self._data[key]['inputs']

    def get_keys(self) -> list[str]:
        """Get the names of all the registered modules

        :return: The list of all the registered modules names
        """
        return self._data.keys()

    def get(self, key: str, **kwargs) -> SPSFGenerator | SDeconvFilter:
        """Get an instance of the SDeconv module

        :param key: Name of the module to load
        :param kwargs: Dictionary of args for models parameters (ex: number of channels)
        :return: The instance of the module
        """
        metadata = self._data.get(key)
        if not metadata:
            raise ValueError(key)
        builder = SDeconvModuleBuilder()
        return builder.get_instance(metadata, kwargs)


class SDeconvModuleBuilder:
    """Interface for a SDeconv module builder

    The builder is used by the factory to instantiate a module

    """
    def __init__(self):
        self._instance = None

    def get_instance(self, metadata: dict[str, any], args: dict) -> SPSFGenerator | SDeconvFilter:
        """Get the instance of the module

        :param metadata: Metadata of the module
        :param args: Argument to pass for the module instantiation
        :return: Instance of the module
        """
        # check the args
        instance_args = {}
        for key, value in metadata['inputs'].items():
            val = self._get_arg(value, key, args)
            instance_args[key] = val
        return metadata['class'](**instance_args)

    def _get_arg(self, param_metadata: dict[str, any], key: str, args: dict[str, any]) -> any:
        """Retrieve the value of a parameter with a type check
        
        :param param_metadata: Metadata of the parameter,
        :param key: Name of the parameter,
        :param args: Value of the parameters
        :return: The value of the parameter if check is successful
        """
        type_ = param_metadata['type']
        range_ = None
        if 'range' in param_metadata:
            range_ = param_metadata['range']
        arg_value = None
        if type_ == 'float':
            arg_value = self.get_arg_float(args, key, param_metadata['default'],
                                           range_)
        elif type_ == 'int':
            arg_value = self.get_arg_int(args, key, param_metadata['default'],
                                         range_)
        elif type_ == 'bool':
            arg_value = self.get_arg_bool(args, key, param_metadata['default'],
                                          range_)
        elif type_ == 'str':
            arg_value = self.get_arg_str(args, key, param_metadata['default'])
        elif type_ is torch.Tensor:
            arg_value = self.get_arg_array(args, key, param_metadata['default'])
        elif type_ == 'select':
            arg_value = self.get_arg_select(args, key, param_metadata['values'])
        elif 'zyx' in type_:
            arg_value = self.get_arg_list(args, key, param_metadata['default'])
        return arg_value

    @staticmethod
    def _error_message(key: str, value_type: str, value_range: tuple | None):
        """Throw an exception if an input parameter is not correct
        
        :param key: Input parameter key
        :param value_type: String naming the input type (int, float...)
        :param value_range: Min and max values of the parameter
        """
        range_message = ''
        if value_range and len(value_range) == 2:
            range_message = f' in range [{str(value_range[0]), str(value_range[1])}]'

        message = f'Parameter {key} must be of type `{value_type}` {range_message}'
        return message

    def get_arg_int(self,
                    args: dict[str, any],
                    key: str,
                    default_value: int,
                    value_range: tuple = None
                    ) -> int:
        """Get the value of a parameter from the args list
        
        The default value of the parameter is returned if the
        key is not in args
        
        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :param value_range: Min and max value of the parameter
        :return: The arg value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = int(args[key])
            except ValueError as exc:
                raise SDeconvFactoryError(self._error_message(key, 'int', value_range)) from exc
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SDeconvFactoryError(self._error_message(key, 'int', value_range))
        return value

    def get_arg_float(self,
                      args: dict[str, any],
                      key: str,
                      default_value: float,
                      value_range: tuple = None
                      ) -> str:
        """Get the value of a parameter from the args list
        
        The default value of the parameter is returned if the
        key is not in args
        
        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :param value_range: Min and max value of the parameter
        :return: The arg value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = float(args[key])
            except ValueError as exc:
                raise SDeconvFactoryError(self._error_message(key, 'float', value_range)) from exc
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SDeconvFactoryError(self._error_message(key, 'float', value_range))
        return value

    def get_arg_str(self,
                    args: dict[str, any],
                    key: str,
                    default_value: str,
                    value_range: tuple = None
                    ) -> str:
        """Get the value of a parameter from the args list
        
        The default value of the parameter is returned if the
        key is not in args
        
        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :param value_range: Min and max value of the parameter
        :return: The arg value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            # cast
            try:
                value = str(args[key])
            except ValueError as exc:
                raise SDeconvFactoryError(self._error_message(key, 'str', value_range)) from exc
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SDeconvFactoryError(self._error_message(key, 'str', value_range))
        return value

    @staticmethod
    def _str2bool(value: str) -> bool:
        """Convert a string to a boolean
        
        :param value: String to convert
        :return: The boolean conversion
        """
        return value.lower() in ("yes", "true", "t", "1")

    def get_arg_bool(self,
                     args: dict[str, any],
                     key: str,
                     default_value: bool,
                     value_range: tuple = None
                     ) -> bool:
        """Get the value of a parameter from the args list
        
        The default value of the parameter is returned if the
        key is not in args
        
        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :param value_range: Min and max value of the parameter
        :return: The arg value
        """
        value = default_value
        # cast
        if isinstance(args, dict) and key in args:
            if isinstance(args[key], str):
                value = SDeconvModuleBuilder._str2bool(args[key])
            elif isinstance(args[key], bool):
                value = args[key]
            else:
                raise SDeconvFactoryError(self._error_message(key, 'bool', value_range))
        # test range
        if value_range and len(value_range) == 2:
            if value > value_range[1] or value < value_range[0]:
                raise SDeconvFactoryError(self._error_message(key, 'bool', value_range))
        return value

    def get_arg_array(self,
                      args: dict[str, any],
                      key: str,
                      default_value: torch.Tensor
                      ) -> torch.Tensor:
        """Get the value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args

        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :return: The arg value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            if isinstance(args[key], torch.Tensor):
                value = args[key]
            elif isinstance(args[key], np.ndarray):
                value = torch.Tensor(args[key])
            else:
                raise SDeconvFactoryError(self._error_message(key, 'array', None))
        return value

    def get_arg_list(self,
                     args: dict[str, any],
                     key: str,
                     default_value: list
                     ) -> list:
        """Get the value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args

        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param default_value: Default value of the parameter
        :return: The arg value
        """
        value = default_value
        if isinstance(args, dict) and key in args:
            if isinstance(args[key], (list, tuple)):
                value = args[key]
            else:
                raise SDeconvFactoryError(self._error_message(key, 'list', None))
        return value

    def get_arg_select(self,
                       args: dict[str, any],
                       key: str,
                       values: list
                       ) -> str:
        """Get the value of a parameter from the args list as a select input

        :param args: Dictionary of the input args
        :param key: Name of the parameters
        :param values: Possible values in select input
        :return: The arg value
        """
        if isinstance(args, dict) and key in args:
            value = str(args[key])
            for val in values:
                if str(val) == value:
                    return val
        raise SDeconvFactoryError(self._error_message(key, 'select', None))
