"""Implements factory for PSF and deconvolution modules

Classes
-------
SDeconvFactoryError
SDeconvModuleFactory
SDeconvModuleBuilder

"""

import numpy as np
import torch


class SDeconvFactoryError(Exception):
    """Raised when an error happen when a module is built in the factory"""


class SDeconvModuleFactory:
    """Factory for SDeconv modules"""
    def __init__(self):
        self._data = {}

    def register(self, key, metadata):
        """Register a new builder to the factory

        Parameters
        ----------
        key: str
            Name of the module to register
        metadata: dict
            Dictionary containing the filter metadata

        """
        self._data[key] = metadata

    def get_parameters(self, key):
        """Parameters getter method"""
        return self._data[key]['inputs']

    def get_keys(self):
        """Get the names of all the registered modules

        Returns
        -------
        list: list of all the registered modules names

        """
        return self._data.keys()

    def get(self, key, **kwargs):
        """Get an instance of the SDeconv module

        Parameters
        ----------
        key: str
            Name of the module to load
        kwargs: dict
            Dictionary of args for models parameters (ex: number of channels)

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

    def get_instance(self, metadata, args):
        """Get the instance of the module

        Returns
        -------
        Object: instance of the SDeep module

        """
        # check the args
        instance_args = {}
        for key, value in metadata['inputs'].items():
            val = self._get_arg(value, key, args)
            instance_args[key] = val
        return metadata['class'](**instance_args)

    def _get_arg(self, param_metadata, key, args):
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
    def _error_message(key, value_type, value_range):
        """Throw an exception if an input parameter is not correct
        Parameters
        ----------
        key: str
            Input parameter key
        value_type: str
            String naming the input type (int, float...)
        value_range: tuple or None
            Min and max values of the parameter
        """
        range_message = ''
        if value_range and len(value_range) == 2:
            range_message = f' in range [{str(value_range[0]), str(value_range[1])}]'

        message = f'Parameter {key} must be of type `{value_type}` {range_message}'
        return message

    def get_arg_int(self, args, key, default_value, value_range=None):
        """Get the value of a parameter from the args list
        The default value of the parameter is returned if the
        key is not in args
        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: int
            Default value of the parameter
        value_range: tuple
            Min and max value of the parameter
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

    def get_arg_float(self, args, key, default_value, value_range=None):
        """Get the value of a parameter from the args list
        The default value of the parameter is returned if the
        key is not in args
        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: float
            Default value of the parameter
        value_range: tuple
            Min and max value of the parameter
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

    def get_arg_str(self, args, key, default_value, value_range=None):
        """Get the value of a parameter from the args list
        The default value of the parameter is returned if the
        key is not in args
        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: str
            Default value of the parameter
        value_range: tuple
            Min and max value of the parameter
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
    def _str2bool(value):
        return value.lower() in ("yes", "true", "t", "1")

    def get_arg_bool(self, args, key, default_value, value_range=None):
        """Get the value of a parameter from the args list
        The default value of the parameter is returned if the
        key is not in args
        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: bool
            Default value of the parameter
        value_range: tuple
            Min and max value of the parameter
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

    def get_arg_array(self, args, key, default_value):
        """Get the value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args

        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: np.array
            Default value of the parameter

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

    def get_arg_list(self, args, key, default_value):
        """Get the value of a parameter from the args list

        The default value of the parameter is returned if the
        key is not in args

        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        default_value: np.array
            Default value of the parameter

        """
        value = default_value
        if isinstance(args, dict) and key in args:
            if isinstance(args[key], list) or isinstance(args[key], tuple):
                value = args[key]
            else:
                raise SDeconvFactoryError(self._error_message(key, 'list', None))
        return value

    def get_arg_select(self, args, key, values):
        """Get the value of a parameter from the args list as a select input

        Parameters
        ----------
        args: dict
            Dictionary of the input args
        key: str
            Name of the parameters
        values: list
            Possible values in select input

        """
        if isinstance(args, dict) and key in args:
            value = str(args[key])
            for val in values:
                if str(val) == value:
                    return val
        raise SDeconvFactoryError(self._error_message(key, 'select', None))
