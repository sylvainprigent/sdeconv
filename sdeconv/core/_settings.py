import torch


class SSettingsContainer:
    """Container for the SDeconv library settings"""
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SSettings:
    """Singleton to access the Settings container
        
    Raises
    ------
    Exception: if multiple instantiation of the Settings container is tried

    """
    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if SSettings.__instance is not None:
            raise Exception("Settings container can be initialized only once!")
        else:
            SSettings.__instance = SSettingsContainer()

    @staticmethod
    def instance():
        """ Static access method to the Config. """
        if SSettings.__instance is None:
            SSettings.__instance = SSettingsContainer()
        return SSettings.__instance
