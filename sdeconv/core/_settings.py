"""Implements setting management"""
import torch


class SSettingsContainer:
    """Container for the SDeconv library settings"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device(self) -> str:
        """Returns the device name for torch"""
        return self.device

    def print(self):
        """Display the settings in the console"""
        print(f'SDeconv settings: device={self.device}')


class SSettings:
    """Singleton to access the Settings container

    :raises: Exception: if multiple instantiation of the Settings container is tried
    """
    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if SSettings.__instance is not None:
            raise RuntimeError("Settings container can be initialized only once!")
        SSettings.__instance = SSettingsContainer()

    @staticmethod
    def instance():
        """ Static access method to the Config. """
        if SSettings.__instance is None:
            SSettings.__instance = SSettingsContainer()
        return SSettings.__instance

    @staticmethod
    def print():
        """Print the settings to the console"""
        SSettings.instance().print()
