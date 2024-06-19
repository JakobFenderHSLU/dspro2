import logging


def init_logging(name: str, level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)-15s - [%(filename)s:%(lineno)d]: %(message)s')

    fh = logging.FileHandler(f'logs/app.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log
