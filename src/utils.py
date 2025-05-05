import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configures basic logging.
    Uses force=True to override any existing root logger configuration.
    """
    log_format = '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s' 
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout) # force=True to enable full debug
    #logging.info(f"Logging configured with level={logging.getLevelName(level)} (Forced)")

    logging.getLogger("ultralytics").setLevel(logging.WARNING)