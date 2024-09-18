"""Core module for the SDeconv library. It implements settings and observer/observable design
patterns"""
from ._observers import SObservable, SObserver, SObserverConsole
from ._settings import SSettings, SSettingsContainer
from ._timing import seconds2str
from ._progress_logger import SConsoleLogger


__all__ = ['SSettings',
           'SSettingsContainer',
           'SObservable',
           'SObserver',
           'SObserverConsole',
           'seconds2str',
           'SConsoleLogger']
