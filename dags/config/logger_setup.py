import logging.config, logging.handlers
from logging.handlers import RotatingFileHandler
import pathlib
import yaml

logger = logging.getLogger(__name__)

def setup_logging():
    ''' Sets up the logger according to the configuration file. '''

    config_path = pathlib.Path(__file__).parent / 'log_config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)