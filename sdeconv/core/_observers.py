"""Module that implements the observer/observable design pattern to display progress"""
from abc import ABC, abstractmethod


class SObserver(ABC):
    """Interface of observer to notify progress

    An observer must implement the progress and message
    """
    @abstractmethod
    def notify(self, message: str):
        """Notify a progress message

        :param message: Progress message
        """
        raise NotImplementedError('SObserver is abstract')

    @abstractmethod
    def progress(self, value: int):
        """Notify progress value

        :param value: Progress value in [0, 100]
        """
        raise NotImplementedError('SObserver is abstract')


class SObservable:
    """Interface for data processing class

    The observable class can notify the observers for progress
    """
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: SObserver):
        """Add an observer

        :param observer: Observer instance to add
        """
        self._observers.append(observer)

    def notify(self, message: str):
        """Notify progress to observers

        :param message: Progress message
        """
        for obs in self._observers:
            obs.notify(message)

    def progress(self, value):
        """Notify progress to observers

        :param value: Progress value in [0, 100]
        """
        for obs in self._observers:
            obs.progress(value)


class SObserverConsole(SObserver):
    """print message and progress to console"""

    def notify(self, message: str):
        """Print message

        :param message: Progress message
        """
        print(message)

    def progress(self, value: str):
        """Print progress

        :param value: Progress value in [0, 100]
        """
        print('progress:', value, '%')
