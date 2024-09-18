"""Set of classes to log a workflow run"""
COLOR_WARNING = '\033[93m'
COLOR_ERROR = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_ENDC = '\033[0m'


class SProgressLogger:
    """Default logger

    A logger is used by a workflow to print the warnings, errors and progress.
    A logger can be used to print in the console or in a log file

    """
    def __init__(self):
        self.prefix = ''

    def new_line(self):
        """Print a new line in the log"""
        raise NotImplementedError()

    def message(self, message: str):
        """Log a default message

        :param message: Message to log
        """
        raise NotImplementedError()

    def error(self, message: str):
        """Log an error message

        :param message: Message to log
        """
        raise NotImplementedError()

    def warning(self, message: str):
        """Log a warning

        :param message: Message to log
        """
        raise NotImplementedError()

    def progress(self, iteration: int, total: int, prefix: str, suffix: str):
        """Log a progress

        :param iteration: Current iteration
        :param total: Total number of iteration
        :param prefix: Text to print before the progress
        :param suffix: Text to print after the message
        """
        raise NotImplementedError()

    def close(self):
        """Close the logger"""
        raise NotImplementedError()


class SProgressObservable:
    """Observable pattern

    This pattern allows to set multiple progress logger to
    one workflow

    """
    def __init__(self):
        self._loggers = []

    def set_prefix(self, prefix: str):
        """Set the prefix for all loggers

        The prefix is a printed str ad the beginning of each
        line of the logger

        :param prefix: Prefix content
        """
        for logger in self._loggers:
            logger.prefix = prefix

    def add_logger(self, logger: SProgressLogger):
        """Add a logger to the observer

        :param logger: Logger to add to the observer
        """
        self._loggers.append(logger)

    def new_line(self):
        """Print a new line in the loggers"""
        for logger in self._loggers:
            logger.new_line()

    def message(self, message: str):
        """Log a default message

        :param message: Message to log
        """
        for logger in self._loggers:
            logger.message(message)

    def error(self, message: str):
        """Log an error message

        :param message: Message to log
        """
        for logger in self._loggers:
            logger.error(message)

    def warning(self, message: str):
        """Log a warning message

        :param message: Message to log
        """
        for logger in self._loggers:
            logger.warning(message)

    def progress(self, iteration: int, total: int, prefix: str, suffix: str):
        """Log a progress

        :param iteration: Current iteration
        :param total: Total number of iteration
        :param prefix: Text to print before the progress
        :param suffix: Text to print after the message
        """
        for logger in self._loggers:
            logger.progress(iteration, total, prefix, suffix)

    def close(self):
        """Close the loggers"""
        for logger in self._loggers:
            logger.close()


class SConsoleLogger(SProgressLogger):
    """Console logger displaying a progress bar

    The progress bar display the basic information of a batch loop (loss,
    batch id, time/remaining time)

    """
    def __init__(self):
        super().__init__()
        self.decimals = 1
        self.print_end = "\r"
        self.length = 100
        self.fill = '█'

    def new_line(self):
        print(f"{self.prefix}:\n")

    def message(self, message):
        print(f'{self.prefix}: {message}')

    def error(self, message):
        print(f'{COLOR_ERROR}{self.prefix} ERROR: '
              f'{message}{COLOR_ENDC}')

    def warning(self, message):
        print(f'{COLOR_WARNING}{self.prefix} WARNING: '
              f'{message}{COLOR_ENDC}')

    def progress(self, iteration, total, prefix, suffix):
        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (iteration / float(total)))
        filled_length = int(self.length * iteration // total)
        bar_ = self.fill * filled_length + ' ' * (self.length - filled_length)
        print(f'\r{prefix} {percent}% |{bar_}| {suffix}',
              end=self.print_end)

    def close(self):
        pass
