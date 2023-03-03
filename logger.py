import logging
import sys

# List of loggers, that must not to work with tracking
# loggers_blacklist = ['urllib3']


def get_logger(logger_name: str, level: int) -> logging.Logger:

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create the logging file handler
    fh = logging.FileHandler("log.txt")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create the logging stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# def pop_loggers_blacklist(logger: logging.Logger) -> dict:
#     logger_dict = logger.manager.loggerDict
#     popped_loggers = {}
#     for lgr in loggers_blacklist:
#         if lgr in logger_dict.keys():
#             popped_loggers[lgr] = logger_dict[lgr]
#             logger_dict.pop(lgr)
#     return popped_loggers
#
#
# def push_loggers_blacklist(logger: logging.Logger, loggers_to_push: dict):
#     logger.manager.loggerDict.update(loggers_to_push)
