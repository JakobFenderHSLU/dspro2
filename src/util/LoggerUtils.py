import logging


def init_logging(name: str, level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(f'logs/app.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log
