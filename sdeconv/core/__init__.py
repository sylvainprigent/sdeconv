"""Core module for the SDeconv library. It implements settings and observer/observable design
patterns"""
from ._settings import SSettings, SSettingsContainer
from ._observers import SObservable, SObserver, SObserverConsole


__all__ = ['SSettings', 'SSettingsContainer',
           'SObservable', 'SObserver', 'SObserverConsole']
