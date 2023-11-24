import configparser
import os
from pathlib import Path
from threading import local
from urllib.parse import urlparse

SKLEARN = "sklearn"
KERAS = "keras"
TENSORFLOW = "tensorflow"
PYTORCH = "pytorch"
CUSTOM = "custom"

DEBUG = False
CONFIG_FILE_PATH = f"{Path.home()}/.baseten_config"

thread_data = local()


def is_config_file_valid() -> bool:
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_PATH)
        # read() doesn't raise an exception if the file doesn't exist
        return os.path.exists(CONFIG_FILE_PATH)
    except configparser.ParsingError:
        return False


def set_config_defaults():
    config = configparser.ConfigParser()
    if is_config_file_valid():
        config.read(CONFIG_FILE_PATH)
    if not config.has_section("api"):
        config.add_section("api")
    config.set("api", "url_base", "https://app.baseten.co")
    with open(CONFIG_FILE_PATH, "w") as configfile:
        config.write(configfile)

    return config


def read_config():
    if not is_config_file_valid():
        set_config_defaults()
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    return config


def set_config_value(section: str, key: str, value: str):
    config = read_config()
    if section not in config.sections():
        config.add_section(section)
    config.set(section, key, value)
    with open(CONFIG_FILE_PATH, "w") as configfile:
        config.write(configfile)


def get_server_url() -> str:
    config = read_config()
    try:
        return config.get("api", "url_base")
    except (configparser.NoSectionError, configparser.NoOptionError):
        config = set_config_defaults()
        return config.get("api", "url_base")


def _is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def set_jwt_auth_token(token: str):
    thread_data.jwt_auth_token = token


def get_jwt_auth_token():
    if hasattr(thread_data, "jwt_auth_token"):
        return getattr(thread_data, "jwt_auth_token")
    return None


def set_server_url(server_url):
    if not _is_valid_url(server_url):
        return False
    set_config_value("api", "url_base", server_url.strip("/"))
    return True
