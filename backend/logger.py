import logging
import sys

from backend.settings import settings

logger = logging.getLogger(__name__)
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def setup_logging(level):
    logger.setLevel(logging.DEBUG)
    log_level = logging.getLevelName(level.upper())
    formatter = logging.Formatter(
        "%(levelname)s:    %(asctime)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


setup_logging(level=settings.LOG_LEVEL)
