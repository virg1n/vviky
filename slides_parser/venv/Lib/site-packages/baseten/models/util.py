import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

import cloudpickle
import requests

from baseten.common import api
from baseten.common.core import AuthorizationError

MODEL_ZOO_CONFIG = {
    "Stable Diffusion": "stable_diffusion",
    "Whisper": "whisper",
    "Flan-T5 XL": "flan_t5",
}


def from_str_to_object(inp: str) -> Any:
    """Converts a hex string that represents a Cloudpickled Python object to a Python object.

    Args:
        inp (str): Hex string that represents a Cloudpickled Python object

    Returns:
        Any: Python object
    """
    byte_stream = bytes.fromhex(inp)
    return cloudpickle.loads(byte_stream)


def from_object_to_str(inp) -> str:
    """Converts a Python object to a hex string that represents a Cloudpickled Python object.

    Args:
        inp (Any): Python object

    Returns:
        str: Hex string that represents a Cloudpickled Python object
    """
    return cloudpickle.dumps(inp).hex()


def get_pretrained_model(model_name: str) -> Optional[Dict]:
    """Checks a users pretrained models models to see if they have a specific model.
    Models are also checked to be in healthy condition. If they are not, we do not
    return them.

    Healthy models are 1 of 2 cases:
    1. The model has a primary version and that primary version is healthy. If a primary
    version is not healthy, we do not return the model. If a primary version exists,
    Django will route all requests to it by default so we need to make sure it is healthy.
    2. The model does not have a primary version, but it has a healthy version.

    Args:
        model_name (str): Name of the model to check for

    Returns:
        Optional[Dict]: Model version if model exists, else None
    """
    try:
        pretrained_model_version_api_response = api.pretrained_model_version(
            pretrained_model_name=MODEL_ZOO_CONFIG[model_name]
        )
    except AuthorizationError:
        raise AuthorizationError("You must first run the `baseten signup` command")
    # See if the model exists in users models

    pretrained_model_version = pretrained_model_version_api_response.get("pretrained_model_version")

    if pretrained_model_version:
        status = pretrained_model_version["current_deployment_status"]
        if status in ["MODEL_READY", "DEPLOYING_MODEL"]:
            return pretrained_model_version
        # If a model is set as primary and is not ready, Django will route requests to the
        # failed model so we need to create a new model version so we return None.
        return None

    return None


def create_pretrained_model(model_name: str) -> dict:
    """Creates a model from the model zoo and returns the model id.

    Args:
        model_name (str): Name of the model to create

    Returns:
        dict: model
    """
    return api.create_pretrained_model(MODEL_ZOO_CONFIG[model_name], model_name)


def get_or_create_pretrained_model(model_name: str) -> Dict:
    """Checks if a model exists in the users models. If it does not, it creates it.

    Args:
        model_name (str): Name of the model to check for

    Returns:
        str: Model id
    """
    if model_name not in MODEL_ZOO_CONFIG:
        raise ValueError("Model {model_name} not supported")
    with requests_error_handling():
        model_version = (
            get_pretrained_model(model_name)
            or create_pretrained_model(model_name)["primaryVersion"]
        )

    return model_version


def upload_file_to_s3(file_path: str) -> str:
    """Uploads a file to S3 and returns the S3 url.

    Args:
        file_path (str): Path to file to upload

    Returns:
        str: The URL to get that file from
    """
    file_extension = file_path.split(".")[-1]
    temp_data_name = f"{str(uuid.uuid4())}.{file_extension}"

    signed_s3_upload_post = api.signed_s3_upload_and_download_post(temp_data_name)
    file_data = {"file": (temp_data_name, open(file_path, "rb"))}

    form_fields = signed_s3_upload_post["form_fields"]
    form_fields["AWSAccessKeyId"] = form_fields.pop("aws_access_key_id")

    with requests_error_handling():
        requests.post(signed_s3_upload_post["url"], data=form_fields, files=file_data)
    return signed_s3_upload_post["get_url"]


@contextmanager
def requests_error_handling():
    """Context manager to handle Requests errors"""
    try:
        yield
    except requests.exceptions.HTTPError as errh:
        print("ERROR: We encountered an HTTP error: ", errh)
    except requests.exceptions.ConnectionError as errc:
        print("ERROR: We are having trouble connecting:", errc)
    except requests.exceptions.Timeout as errt:
        print("ERROR: The request to the model has timed out:", errt)
    except requests.exceptions.RequestException as err:
        print("ERROR: Something went wrong, please try again:", err)
