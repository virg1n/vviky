import functools
import json
import logging
import os
from typing import IO, Dict, List, Optional, Tuple, Union

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from colorama import Fore
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm

from baseten.common import settings
from baseten.common.core import ApiError, AuthorizationError
from baseten.common.util import base64_encoded_json_str

logger = logging.getLogger(__name__)


class AuthToken:
    value: str

    def __init__(self, value: str):
        self.value = value

    def headers(self):
        raise NotImplementedError


class ApiKey(AuthToken):
    def headers(self):
        return {"Authorization": f"Api-Key {self.value}"}


class JWT(AuthToken):
    def headers(self):
        return {"Authorization": self.value}


def with_api_key(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = settings.read_config()

        if config.has_option("api", "api_key"):
            api_key = config.get("api", "api_key")
        elif settings.get_jwt_auth_token():
            raise AuthorizationError("This operation is not supported in the Web IDE")
        else:
            raise AuthorizationError("You must first run the `baseten login` cli command.")

        result = func(ApiKey(api_key), *args, **kwargs)
        return result

    return wrapper


def with_api_key_or_jwt(func):
    """
    Allows an API method to be called internally (with JWT) or externally
    (with API Key).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = settings.read_config()

        if config.has_option("api", "api_key"):
            result = func(ApiKey(config.get("api", "api_key")), *args, **kwargs)
            return result
        elif settings.get_jwt_auth_token():
            result = func(JWT(settings.get_jwt_auth_token()), *args, **kwargs)
            return result
        else:
            raise AuthorizationError("You must first run the `baseten login` cli command.")

    return wrapper


@with_api_key_or_jwt
def models(auth_token: AuthToken):
    query_string = """
    {
      models {
        id,
        name
        versions{
            id,
            semver,
            current_deployment_status,
            is_primary,
        }
      }
    }
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]


@with_api_key
def get_model(api_key, model_name):
    query_string = f"""
    {{
      model_version(name: "{model_name}") {{
        oracle{{
            name
            versions{{
                id
                semver
                truss_hash
                truss_signature
                is_draft
                current_model_deployment_status {{
                    status
                }}
            }}
        }}
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]


@with_api_key_or_jwt
def get_primary_model_version_status(api_key, model_id: str):
    query_string = f"""
    {{
      model(id: "{model_id}") {{
        primary_version{{
            current_model_deployment_status {{
                status
            }}
        }}
        }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model"]["primary_version"]["current_model_deployment_status"]["status"]


@with_api_key_or_jwt
def get_model_version_status(api_key, model_version_id: str):
    query_string = f"""
    {{
        model_version(id: "{model_version_id}") {{
            current_model_deployment_status {{
                status
            }}
        }}
    }}

    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model_version"]["current_model_deployment_status"]["status"]


@with_api_key
def get_pretrained_model(api_key, pretrained_model_name):
    query_string = f"""
    {{
      pretrained_model(name: "{pretrained_model_name}") {{
        pretty_name
        name
        s3_key
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]


@with_api_key
def pretrained_models(api_key):
    query_string = """
    {
      pretrained_models {
        pretty_name
        name
        s3_key
      }
    }
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]


@with_api_key_or_jwt
def pretrained_model_version(auth_token: AuthToken, pretrained_model_name: str):
    query_string = f"""
    {{
      pretrained_model_version(pretrained_model_name: "{pretrained_model_name}"){{
        model_id
        current_deployment_status
        truss_spec_version
      }}
    }}
    """
    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]


@with_api_key
def models_summary(api_key):
    query_string = """
    {
      models {
        id,
        name,
        description,
        versions {
          id
          created
        }
      }
    }
    """

    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["models"]


@with_api_key
def model_version_external_id_get_version(api_key, external_model_version_id):
    query_string = f"""
    {{
      model_version(external_model_version_id: "{external_model_version_id}") {{
        id
      }}
    }}
    """

    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model_version"]["id"]


@with_api_key_or_jwt
def model_version_truss_spec_version(auth_token: AuthToken, model_version_id: str):
    query_string = f"""
    {{
      model_version(id: "{model_version_id}") {{
        truss_spec_version
      }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["model_version"]["truss_spec_version"]


@with_api_key_or_jwt
def model_truss_spec_version(auth_token: AuthToken, model_id: str):
    query_string = f"""
    {{
      model(id: "{model_id}") {{
        primary_version {{
            truss_spec_version
        }}
      }}
    }}
    """
    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["model"]["primary_version"]["truss_spec_version"]


@with_api_key_or_jwt
def model_id_from_model_version(auth_token: AuthToken, model_version_id: str) -> str:
    query_string = f"""
    {{
      model_version(id: "{model_version_id}") {{
        oracle {{
            id
        }}
      }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["model_version"]["oracle"]["id"]


@with_api_key
def create_model_from_truss(
    api_key,
    model_name,
    s3_key,
    config,
    semver_bump,
    client_version,
    is_trusted=False,
    external_model_version_id=None,
):
    query_string = f"""
    mutation {{
      create_model_from_truss(name: "{model_name}",
                   s3_key: "{s3_key}",
                   config: "{config}",
                   semver_bump: "{semver_bump}",
                   client_version: "{client_version}",
                   is_trusted: {'true' if is_trusted else 'false'}
                   external_model_version_id: "{external_model_version_id if external_model_version_id else ''}"
) {{
        id,
        name,
        version_id
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["create_model_from_truss"]


@with_api_key
def create_model_version_from_truss(
    api_key,
    model_id,
    s3_key,
    config,
    semver_bump,
    client_version,
    is_trusted=False,
    external_model_version_id=None,
):
    query_string = f"""
    mutation {{
      create_model_version_from_truss(
                   model_id: "{model_id}"
                   s3_key: "{s3_key}",
                   config: "{config}",
                   semver_bump: "{semver_bump}",
                   client_version: "{client_version}",
                   is_trusted: {'true' if is_trusted else 'false'}
                   external_model_version_id: "{external_model_version_id if external_model_version_id else ''}"

      ) {{
        id,
        name,
        version_id
      }}
    }}
    """

    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["create_model_version_from_truss"]


@with_api_key
def deploy_draft_truss(
    api_key,
    model_name,
    s3_key,
    config,
    client_version,
    is_trusted=False,
):
    query_string = f"""
    mutation {{
      deploy_draft_truss(name: "{model_name}",
                   s3_key: "{s3_key}",
                   config: "{config}",
                   client_version: "{client_version}",
                   is_trusted: {'true' if is_trusted else 'false'},
) {{
        id,
        name,
        version_id
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["deploy_draft_truss"]


@with_api_key
def patch_draft_truss(
    api_key,
    model_name,
    patch,
    client_version,
):
    patch = base64_encoded_json_str(patch.to_dict())
    query_string = f"""
    mutation {{
      patch_draft_truss(name: "{model_name}",
                   client_version: "{client_version}",
                   patch: "{patch}",
) {{
        id,
        name,
        version_id
        succeeded
        needs_full_deploy
        error
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["patch_draft_truss"]


@with_api_key
def signed_s3_upload_post(api_key, model_file_name):
    query_string = f"""
    {{
      signed_s3_upload_url(model_file_name: "{model_file_name}") {{
        url,
        form_fields {{
          key,
          aws_access_key_id,
          policy,
          signature,
        }}
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["signed_s3_upload_url"]


@with_api_key
def model_s3_upload_credentials(api_key):
    query_string = """
    {
        model_s3_upload_credentials {
            s3_bucket
            s3_key
            aws_access_key_id
            aws_secret_access_key
            aws_session_token
        }
    }
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model_s3_upload_credentials"]


@with_api_key
def signed_s3_upload_and_download_post(api_key, file_name):
    query_string = f"""
    {{
      signed_s3_upload_url(model_file_name: "{file_name}"){{
        url,
        get_url,
        form_fields {{
          key,
          aws_access_key_id,
          policy,
          signature,
        }}
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["signed_s3_upload_url"]


@with_api_key
def get_model_s3_download_url(api_key, model_version_id):
    query_string = f"""
    {{
      model_s3_download_url(model_version_id: "{model_version_id}") {{
        url,
      }}
    }}
    """

    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model_s3_download_url"]["url"]


@with_api_key_or_jwt
def user_files_signed_s3_upload_post(auth_token: AuthToken, file_name: str):
    query_string = f"""
    {{
      user_files_signed_s3_upload_url(file_name: "{file_name}") {{
        url,
        get_url,
        form_fields {{
          key,
          aws_access_key_id,
          policy,
          signature,
        }}
      }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)

    return resp["data"]["user_files_signed_s3_upload_url"]


@with_api_key
def register_data_for_model(api_key, s3_key, model_version_id, data_name):
    query_string = f"""
    mutation {{
        create_sample_data_file(model_version_id: "{model_version_id}",
                                name: "{data_name}",
                                s3_key: "{s3_key}") {{
          id
        }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["create_sample_data_file"]


@with_api_key_or_jwt
def predict_for_model(
    auth_token: AuthToken,
    model_id: str,
    inputs: Union[List, Dict],
) -> Union[List[List], Dict]:
    """Call the model's predict given the input json.

    Args:
        auth_token (AuthToken)
        model_id (str)
        inputs (List)

    Returns
        The model output

    Raises:
        RequestException: If there was an error communicating with the server.
    """

    # We have a special predict endpoint for internal use that we use
    # if a JWT is present.
    if isinstance(auth_token, ApiKey):
        predict_url = f"{settings.get_server_url()}/models/{model_id}/predict"
    elif isinstance(auth_token, JWT):
        predict_url = f"{settings.get_server_url()}/models/{model_id}/predict_internal"

    return _predict(auth_token, predict_url, inputs)


@with_api_key
def predict_for_model_version(
    api_key: ApiKey,
    model_version_id: str,
    inputs: Union[List, Dict],
) -> Union[List[List], Dict]:
    """Call the model version's predict given the input json.

    Args:
        api_key (str)
        model_version_id (str)
        inputs (List|Dict)

    Returns
        The model output

    Raises:
        RequestException: If there was an error communicating with the server.
    """
    predict_url = f"{settings.get_server_url()}/model_versions/{model_version_id}/predict"
    return _predict(api_key, predict_url, inputs)


@with_api_key
def set_primary(api_key, model_version_id: str):
    """Promote this version of the model as the primary version.

    Args:
        api_key (str)
        model_version_id (str)
    """
    query_string = f"""
    mutation {{
      update_model_version(model_version_id: "{model_version_id}", is_primary: true) {{
        id,
        is_primary,
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["update_model_version"]


@with_api_key
def update_model_features(
    api_key: ApiKey, model_version_id: str, feature_names: List, class_labels: Optional[List] = None
):
    """Update the feature names for the model.

    Args:
        api_key (str)
        model_version_id (str)
        feature_names (List)
        class_labels (Optional[List]): applies only to classifiers.
    """
    encoded_feature_names = base64_encoded_json_str(feature_names)
    encoded_class_labels = base64_encoded_json_str(class_labels)
    query_string = f"""
    mutation {{
      update_model_version(model_version_id: "{model_version_id}",
                           encoded_feature_names: "{encoded_feature_names}",
                           encoded_class_labels: "{encoded_class_labels}") {{
        id,
        feature_names,
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["update_model_version"]


@with_api_key
def install_requirements(api_key, requirements_txt):
    escaped_requirements_txt = requirements_txt.replace(
        "\n", "\\n"
    )  # Otherwise the mutation becomes invalid graphql.
    query_string = f"""
    mutation {{
      create_pynode_requirement(requirements_txt: "{escaped_requirements_txt}") {{
        id
        status
        error_message
      }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["create_pynode_requirement"]


@with_api_key
def requirement_status(api_key, requirement_id):
    query_string = f"""
    {{
      pynode_requirement(id: "{requirement_id}") {{
        id
        status
        error_message
      }}
    }}
    """

    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["pynode_requirement"]


def _format_inputs_for_v1(
    auth_token: AuthToken, inputs: Union[List, Dict], metadata: Optional[List[Dict]]
) -> Union[List, Dict]:
    """
    For internal (JWT) token use-cases, we do not need to package up the inputs.
    For external cases, for the v1 model serving path, we expect inputs to be
    provided in the format { inputs: ..., metadata: ... }.
    """
    if isinstance(auth_token, ApiKey):
        return {"inputs": inputs, "metadata": metadata}
    else:
        return inputs


def _predict(
    auth_token: AuthToken,
    predict_url: str,
    inputs: Union[List, Dict],
):
    resp = _post_rest_query(auth_token, predict_url, inputs, stream=True)

    if resp.headers.get("transfer-encoding") == "chunked":
        # Case of streaming response
        return resp.raw.stream()

    resp_json = json.loads(resp.content)
    return resp_json["model_output"]


def _post_graphql_query(auth_token: AuthToken, query_string: str) -> dict:
    headers = auth_token.headers()

    resp = requests.post(
        f"{settings.get_server_url()}/graphql/", data={"query": query_string}, headers=headers
    )
    if not resp.ok:
        logger.error(f"GraphQL endpoint failed with error: {resp.content}")  # type: ignore
        resp.raise_for_status()
    resp_dict = resp.json()
    errors = resp_dict.get("errors")
    if errors:
        raise ApiError(errors[0]["message"], resp)
    return resp_dict


def _post_rest_query(
    auth_token: AuthToken, url: str, post_body_dict: Union[List, Dict], **request_options
):
    headers = auth_token.headers()

    resp = requests.post(url, json=post_body_dict, headers=headers, **request_options)
    resp.raise_for_status()
    return resp


def upload_model(serialize_file: IO, file_ext: str, file_name: str) -> str:
    """Uploads the serialized model to the appropriate environment

    Args:
        serialize_file (file): A file-like object that is the serialized representation of the model object.
        file_ext (str): The file extension for the saved model

    Returns:
        str: The key for the uploaded model

    Raises:
        RequestException: If there was an error communicating with the server.
    """

    temp_credentials_s3_upload = model_s3_upload_credentials()
    s3_key = temp_credentials_s3_upload.pop("s3_key")
    s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
    logger.info("🚀 Uploading model to Baseten 🚀")
    _multipart_upload_boto3(serialize_file.name, s3_bucket, s3_key, temp_credentials_s3_upload)
    return s3_key


def upload_user_file(io_stream: IO, file_name: str) -> str:
    signed_s3_upload_post = user_files_signed_s3_upload_post(file_name)
    logger.debug(f"Signed s3 upload post:\n{json.dumps(signed_s3_upload_post, indent=4)}")

    form_fields = signed_s3_upload_post["form_fields"]
    form_fields["AWSAccessKeyId"] = form_fields.pop(
        "aws_access_key_id"
    )  # S3 expects key name AWSAccessKeyId
    form_fields["file"] = (file_name, io_stream)

    _upload_file(signed_s3_upload_post, form_fields)

    return signed_s3_upload_post["get_url"]


def upload_user_dataset_file(file_path: str, file_name: str) -> dict:
    signed_s3_upload_post = user_files_signed_s3_upload_post(file_name)
    logger.debug(f"Signed s3 upload post:\n{json.dumps(signed_s3_upload_post, indent=4)}")

    form_fields = signed_s3_upload_post["form_fields"]
    form_fields["AWSAccessKeyId"] = form_fields.pop(
        "aws_access_key_id"
    )  # S3 expects key name AWSAccessKeyId
    form_fields["file"] = (file_name, open(file_path, "rb"))

    _upload_file(signed_s3_upload_post, form_fields)

    return signed_s3_upload_post


def _multipart_upload_boto3(file_path, bucket_name, key, credentials):
    s3_resource = boto3.resource("s3", **credentials)
    filesize = os.stat(file_path).st_size

    with tqdm(
        total=filesize,
        desc="Upload",
        unit="B",
        unit_scale=True,
    ) as pbar:
        s3_resource.Object(bucket_name, key).upload_file(
            file_path,
            Config=TransferConfig(
                max_concurrency=10,
                use_threads=True,
            ),
            Callback=pbar.update,
        )


def _upload_file(this_signed_s3_upload_post: dict, form_fields: dict) -> str:
    encoder = MultipartEncoder(fields=form_fields)
    encoder_len = encoder.len
    pbar = tqdm(
        total=encoder_len,
        unit_scale=True,
        bar_format="Upload Progress: {percentage:3.0f}%% |%s{bar:100}%s| {n_fmt}/{total_fmt}"
        % (Fore.BLUE, Fore.RESET),
    )

    def callback(monitor):
        progress = monitor.bytes_read - pbar.n
        pbar.update(progress)

    monitor = MultipartEncoderMonitor(encoder, callback)
    resp = requests.post(
        this_signed_s3_upload_post["url"],
        data=monitor,
        headers={"Content-Type": monitor.content_type},
    )
    resp.raise_for_status()
    pbar.close()
    logger.info("🔮 Upload successful!🔮")

    logger.debug(f"File upload HTTP status code: {resp.status_code} and content:\n{resp.content}")  # type: ignore

    return this_signed_s3_upload_post["form_fields"]["key"]


@with_api_key
def deactivate_model_version(api_key, model_version_id):
    query_string = f"""
    mutation {{
        deactivate_model_version(model_version_id: "{model_version_id}") {{
          ok
        }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["deactivate_model_version"]["ok"]


@with_api_key
def activate_model_version(api_key, model_version_id):
    query_string = f"""
    mutation {{
        activate_model_version(model_version_id: "{model_version_id}") {{
          ok
        }}
    }}
    """
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["activate_model_version"]["ok"]


@with_api_key_or_jwt
def create_pretrained_model(auth_token: AuthToken, model_zoo_name: str, model_name: str):
    query_string = f"""
    mutation {{
        create_pretrained_model(
            model_zoo_name: "{model_zoo_name}",
            name: "{model_name}",
            description : "",
            create_app: false
        ) {{
            model {{
                created
                id
                name
                platformType: model_platform_type
                numberOfVersions: number_of_versions
                description: description
                primaryVersion: primary_version {{
                    id
                    model_id
                    modelFramework: model_framework_display_name
                    truss_spec_version
                }}
            }}
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["create_pretrained_model"]["model"]


def _generate_graphql_params(params: dict) -> str:
    """
    Generates a GraphQL parameter string from a dictionary
    """
    return ",".join([f'{key}: "{value}"' for key, value in params.items()])


@with_api_key_or_jwt
def create_training_dataset(
    auth_token: AuthToken, s3_key: str, training_type: str, dataset_name: Optional[str]
) -> str:
    # Only include the dataset name if it is passed.
    graphql_params = {"training_type": training_type, "s3_key": s3_key}

    if dataset_name is not None:
        graphql_params.update({"name": dataset_name})

    query_string = f"""
    mutation {{
        create_training_dataset(
            {_generate_graphql_params(graphql_params)}
        ) {{
            id
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["create_training_dataset"]["id"]


@with_api_key_or_jwt
def get_training_dataset(auth_token: AuthToken, dataset_id: str) -> dict:
    query_string = f"""
    {{
        training_dataset(id: "{dataset_id}") {{
            name
            id
            presigned_s3_get_url
            training_type
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["training_dataset"]


@with_api_key_or_jwt
def get_training_run_by_id(auth_token: AuthToken, training_run_id: str) -> Dict:
    query_string = f"""
    {{
        training_run(id: "{training_run_id}") {{
            id
            status
            trained_model_name

            variables {{
                key
                value
            }}
            created
            started
            stopped
            truss_name
            latest_oracle_version {{
                id
                oracle {{
                    id
                }}
            }}
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["training_run"]


@with_api_key_or_jwt
def get_training_run_by_name(auth_token: AuthToken, trained_model_name: str) -> Dict:
    query_string = f"""
    {{
        training_run_by_trained_model_name(trained_model_name: "{trained_model_name}") {{
            id
            status
            trained_model_name
            variables {{
                key
                value
            }}
            created
            started
            stopped
            truss_name
            latest_oracle_version {{
                id
                oracle {{
                    id
                }}
            }}
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["training_run_by_trained_model_name"]


@with_api_key_or_jwt
def get_all_training_runs(auth_token: AuthToken) -> List[Dict]:
    query_string = """
    query {
        training_runs {
            id
            trained_model_name
        }
    }
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["training_runs"]


@with_api_key_or_jwt
def finetune_zoo_model(
    auth_token: AuthToken,
    trained_model_name: str,
    train_truss_name: str,
    encoded_variables: str,
    auto_deploy: bool,
) -> str:
    query_string = f"""
    mutation {{
        finetune_model_zoo_model(
            truss_name: "{train_truss_name}",
            encoded_variables: "{encoded_variables}",
            trained_model_name: "{trained_model_name}",
            auto_deploy: {json.dumps(auto_deploy)}
        ) {{
            id
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["finetune_model_zoo_model"]["id"]


@with_api_key_or_jwt
def deploy_from_training_run(
    auth_token: AuthToken,
    training_run_id: str,
    name: str,
    idle_time_minutes: int,
) -> Tuple[str, str]:
    query_string = f"""
    mutation {{
        deploy_from_training_run(
            training_run_id: "{training_run_id}",
            name: "{name}",
            idle_time_minutes: {idle_time_minutes},
        ) {{
            version_id
            id

        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    deploy_data = resp["data"]["deploy_from_training_run"]
    return deploy_data["id"], deploy_data["version_id"]


@with_api_key_or_jwt
def cancel_training_run(
    auth_token: AuthToken,
    training_run_id: str,
) -> bool:
    query_string = f"""
    mutation {{
        cancel_training_run(
            training_run_id: "{training_run_id}",
        ) {{
            cancelled
        }}
    }}
    """

    resp = _post_graphql_query(auth_token, query_string)
    deploy_data = resp["data"]["cancel_training_run"]
    return deploy_data["cancelled"]


@with_api_key_or_jwt
def get_training_logs(
    auth_token: AuthToken,
    training_run_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    direction: Optional[str] = None,
) -> List[dict]:
    query_string = f"""
    query {{
        logs(
            log_type: TRAINING,
            entity_id: "{training_run_id}",
            {f'start: "{start}",' if start is not None else ""}
            {f'end: "{end}",' if end is not None else ""}
            {f'limit: {limit},' if limit is not None else ""}
            {f'direction: "{direction}",' if direction is not None else ""}
        ) {{
            ts
            level
            msg
        }}
    }}
    """
    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["logs"]


@with_api_key_or_jwt
def get_blueprint_projects(
    auth_token: AuthToken,
) -> List[dict]:
    query_string = """
    query {
        blueprint_workflows {
            id
            name
        }
    }
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["blueprint_workflows"]


@with_api_key_or_jwt
def get_organization_credits(auth_token: AuthToken) -> List[dict]:
    query_string = """
    query {
        organization {
            monetary_credit_granted
            monetary_credit_balance
            payment_method_status
        }
    }
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["organization"]


@with_api_key_or_jwt
def get_organization_scalable_to_zero(auth_token: AuthToken) -> bool:
    query_string = """
    query {
        organization {
            scale_new_models_to_zero
        }
    }
    """

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]["organization"]["scale_new_models_to_zero"]
