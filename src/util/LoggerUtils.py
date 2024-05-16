import logging
import sys


def init_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


class CustomFormatter(logging.Formatter):
    """
    From: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    """

    light_blue = "\x1b[96;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[91;20m"
    bold_red = "\x1b[91;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: light_blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
