"""Baseten

    isort:skip_file
"""

from truss.build import load
from pathlib import Path
from single_source import get_version
import logging
from typing import List, DefaultDict


from collections import defaultdict

from baseten.baseten_deployed_model import BasetenDeployedModel, models_summary  # noqa: E402
from baseten.common.model_deployer import (
    build_truss,
    deploy_truss,
    build_and_deploy_truss,
    pull_model,
)
from baseten.common.files import upload_dataset
from baseten.common.util import setup_logger  # noqa: E402
from baseten.common import settings  # noqa: E402


__version__ = get_version(__name__, Path(__file__).parent.parent)


def version():
    return __version__


patched_modules: DefaultDict = defaultdict(lambda: [])


logger = logging.getLogger(__name__)
setup_logger("baseten", logging.INFO)
if settings.DEBUG:
    setup_logger("baseten", logging.DEBUG)
logger.debug(f"Starting the client with the server URL set to {settings.get_server_url()}")

# The Baseten model ID that either a) the user initialized baseten with, or b) was created when deploying a model.
working_model_id = None

deploy = build_and_deploy_truss  # This allows the user to call baseten.deploy(model)
build_truss = build_truss
deploy_truss = deploy_truss
upload = upload_dataset
load_truss = load
pull = pull_model


models_summary = models_summary


def set_log_level(level):
    setup_logger("baseten", level)


def init(baseten_model_id=None):
    """Initialize Baseten

    Args:
        baseten_model_id (str, optional): The Baseten model id to initialize the client with.
        If not provided, a new Baseten model will be created at deploy() time.
    """
    global working_model_id
    working_model_id = baseten_model_id


def deployed_model_id(model_id: str) -> BasetenDeployedModel:
    """Returns a BasetenDeployedModel object for interacting with the model model_id.

    Args:
        model_id (str): The ID of the model

    Returns:
        BasetenDeployedModel (BasetenDeployedModel): An object for interacting with the model
    """
    return BasetenDeployedModel(model_id=model_id)


def deployed_model_version_id(model_version_id: str) -> BasetenDeployedModel:
    """Returns a BasetenDeployedModel object for interacting with the model version model_version_id.

    Args:
        model_version_id (str): The ID of a model version

    Returns:
        BasetenDeployedModel (BasetenDeployedModel): An object for interacting with the model
    """
    return BasetenDeployedModel(model_version_id=model_version_id)


def deployed_external_model_version_id(external_model_version_id: str) -> BasetenDeployedModel:
    """Returns a BasetenDeployedModel object for interacting with the model version external_model_version_id.

    Args:
        external_model_version_id (str): The non-Baseten-generated ID assigned during model deployment

    Returns:
        BasetenDeployedModel (BasetenDeployedModel): An object for interacting with the model
    """
    return BasetenDeployedModel(external_model_version_id=external_model_version_id)


def login(api_key: str):
    """Set the API key for the client.

    Args:
        api_key (str): Your API key
    """
    settings.set_config_value("api", "api_key", api_key)
    logger.info("API key set.")


def set_jwt(jwt: str):
    """Sets a JWT for the client. For internal use only.

    Args:
        jwt (str)
    """

    settings.set_jwt_auth_token(jwt)
    logger.info("JWT set.")


def configure(server_url: str):
    """Configure client to use on-prem hosted Baseten environment.

    !!! note

        The client works out of the box without any need to run this command.
        This command is only required if you have Baseten Enterprise hosted on-prem and is not
        required for most users.

    Args:
        server_url (str): The base URL of the server
    """
    if settings.set_server_url(server_url):
        logger.info("Saved server URL.")
    else:
        logger.info("That is not a valid URL.")


def route(
    path: str,
    is_public: bool = False,
    allowed_domains: List[str] = [],
    allowed_methods: List[str] = ["GET"],
):
    """
    Decorator: transform a Python function in Blueprint into an endpoint.

    **Example:**

    ```python
    @route(path="/some-route")
    def my_route(request):
        ...
    ```

    ```python
    @route(
        path="/public-route",
        is_public=True,
        allowed_domains=["https://example.com", "https://another-example.com"],
        allowed_methods=["GET", "POST"]
    )
    def my_route(request):
        ...
    ```

    Args:
        path: Path to use for the route
        is_public: is this a public endpoint?
            (Public endpoints do not require an API key to request.)
        allowed_domains: Allowed domains prevent CORS issues when invoking public endpoints.
        allowed_methods: Choose to support `'GET'`, `'POST'`, `'PUT'`, and `'DELETE'` methods.
    """

    def route_decorator(func):
        return func

    return route_decorator
