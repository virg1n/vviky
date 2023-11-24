import json
import logging
from typing import Any, Callable, Optional, Tuple

from colorama import Fore, Style
from truss.build import create
from truss.local.local_config_handler import (
    LocalConfigHandler as TrussLocalConfigHandler,
)
from truss.truss_handle import TrussHandle

import baseten
from baseten.baseten_deployed_model import BasetenDeployedModel
from baseten.common import api, settings
from baseten.common.core import MODEL_FILENAME, Semver, raises_api_error
from baseten.common.tar import create_tar_with_progress_bar
from baseten.common.types import TryPatchDraftTrussResponse
from baseten.common.util import base64_encoded_json_str

logger = logging.getLogger(__name__)


def _exists_model(
    model_name: Optional[str], model_id: Optional[str], models_provider: Callable
) -> Optional[Tuple[str, str]]:
    """Checks if an effective model exists for the purpose of deploy or one needs to be created.

    If model name is supplied and a model with that name exists then it's picked up. Otherwise
    working model id in the session is tried.

    Returns id of model if model exists, or None
    """
    models = models_provider()["models"]
    model_id_by_name = {model["name"]: model["id"] for model in models}
    model_name_by_id = {model["id"]: model["name"] for model in models}

    if model_name is not None:
        if model_name in model_id_by_name:
            return model_id_by_name[model_name], model_name
        else:
            logger.warning("Model name not found in deployed models")
            return None

    # No model_name supplied, try to work with working model id
    if model_id is not None:
        if model_id in model_name_by_id:
            return model_id, model_name_by_id[model_id]
        else:
            logger.warning("Working model id not found in deployed models")
            return None

    # No model name or working model
    return None


def build_truss(
    model: Any,
    target_directory: Optional[str] = None,
) -> TrussHandle:
    """
    Builds truss into target directory from in-memory model
    Args:
        model (an in-memory model object): A model object to be deployed (e.g. a keras,
            sklearn, or pytorch model object)
        target_directory (str, optional): The local directory target for the truss.
            Otherwise a temporary directory will be generated
    Returns:
        TrussHandle
    """
    truss = create(model=model, target_directory=target_directory)

    truss_help_messages = f"""
    --------------------------------------------------------------------------------------------
    | {Fore.BLUE} Autogenerating Truss for your model, find more about Truss at https://truss.baseten.co/ |
    --------------------------------------------------------------------------------------------
    You can find your auto generated Truss at {Fore.BLUE} {str(truss.spec.truss_dir.resolve())}
    ---------------------------------------------------------------
    | {Fore.BLUE} Some useful commands to work with your Truss... |
    ---------------------------------------------------------------
    To load your Truss, you can either
    1. Access the Truss handle via the object returned by this function via {Fore.MAGENTA}deployed_model.truss
    2. Load the Truss handle via {Fore.MAGENTA}handle = truss.load('{str(truss.spec.truss_dir.resolve())}')
    To locally run your Truss before deploying, run {Fore.MAGENTA}handle.predict(test_input)
    To add a requirement to your Truss, run {Fore.MAGENTA}handle.add_python_requirement('numpy')
    Visit the docs to learn more about Truss!
    """
    for msg in truss_help_messages.splitlines():
        logger.info(msg.strip())
    return truss


@raises_api_error
def deploy_truss(
    b10_truss: TrussHandle,
    model_name: Optional[str] = None,
    semver_bump: str = Semver.MINOR.value,
    is_trusted: bool = False,
    publish: bool = False,
    external_model_version_id=None,
) -> BasetenDeployedModel:
    """
    Given an existing baseten truss object, deploy it onto the baseten infastructure.

    Args:
        b10_truss: A truss object representing a local baseten_truss directory
        model_name (str, optional): The name of the model to be created, if necessary.
        semver_bump (str, optional): The version bump for this deployment, one of 'MAJOR', 'MINOR', 'PATCH'.
        is_trusted (bool, optional): Whether or not to mark a truss as `trusted` on Baseten.
        publish (bool, optional): Whether or not to mark a truss as `published` on Baseten.
        external_model_version_id (str, optional): If you would rather track model IDs with your own system,
            pass the custom ID here.
    Returns:
        BasetenDeployedModel
    """
    if model_name is None:
        import coolname

        model_name = coolname.generate_slug(2)

    elif not model_name or model_name.isspace():
        raise ValueError("Model Names cannot be empty or spaces")

    # TODO(pankaj) The exists check after generating random name is incorrect.
    # If we happen to generate a random name that's already taken then we will
    # have unexpected behavior.
    model_info = _exists_model(model_name, baseten.working_model_id, api.models)

    if model_info is not None and not publish:
        resp = try_patch_draft_truss(model_name, b10_truss)
        if resp.succeeded or not resp.needs_full_deploy:
            if resp.succeeded:
                logger.info(resp.success_message or f"Model {model_name} patched successfully.")
            else:
                logger.error(
                    f"Failed to patch: `{resp.error_message}`. Model left in original state."
                )

            version_info = resp.model_version_info
            if version_info is None:
                raise RuntimeError("Expecting model version info to be set but found to be None")
            model_id, model_name = model_info
            model_version_id = version_info["id"]
            return BasetenDeployedModel(
                model_version_id=model_version_id, model_name=model_name, truss_handle=b10_truss
            )
        else:
            logger.info(f"Giving up on patching: {resp.error_message}")

    logger.info(f"Serializing {Fore.BLUE}{model_name}{Style.RESET_ALL} truss.")

    # If the truss is not scattered then gather is a no-op and returns path to
    # original truss. So, there's no additional cost to gather for trusses
    # without external packages.
    gathered_truss = TrussHandle(b10_truss.gather())
    temp_file = _archive_truss(b10_truss=gathered_truss)
    logger.info("Making contact with Baseten 👋 👽")
    s3_key = api.upload_model(temp_file, "tgz", MODEL_FILENAME)
    # String that can be passed through graphql api
    config = base64_encoded_json_str(gathered_truss._spec._config.to_dict())
    if not publish:
        model_id, model_version_id = deploy_draft_truss(
            model_name=model_name,
            s3_key=s3_key,
            config=config,
            is_trusted=is_trusted,
        )
    else:
        if not model_info:
            model_id, model_version_id = create_model_from_s3(
                model_name, s3_key, config, semver_bump, is_trusted, external_model_version_id
            )
            logger.info(f"Successfully registered model {Fore.BLUE}{model_name}{Style.RESET_ALL}.")
        else:
            model_id, model_name = model_info
            model_version_id = create_model_version_from_s3(
                model_id, s3_key, config, semver_bump, is_trusted, external_model_version_id
            )
            logger.info(
                f"Successfully created version {Fore.BLUE}{model_version_id}{Style.RESET_ALL} for {model_name}."
            )

    _generate_model_deploy_logs(model_id)
    return BasetenDeployedModel(
        model_version_id=model_version_id, model_name=model_name, truss_handle=b10_truss
    )


def deploy_draft_truss(
    model_name,
    s3_key,
    config,
    is_trusted=False,
):
    model_version_json = api.deploy_draft_truss(
        model_name=model_name,
        s3_key=s3_key,
        config=config,
        client_version=baseten.__version__,
        is_trusted=is_trusted,
    )
    can_scale_to_zero = api.get_organization_scalable_to_zero()
    scaling_message = (
        f"""
    \n📉 {Fore.MAGENTA}By default this model will scale down to zero after one hour without requests,
saving you money. When the model is invoked again it may take some time to start back up.
You can update model resources and autoscaling settings in your Baseten account."""
        if can_scale_to_zero
        else ""
    )

    logger.info(
        f"""
|---------------------------------------------------------------------------------------|
| Your model has been deployed as a draft. Draft models allow you to                    |
| iterate quickly during the deployment process. For more information,                  |
| read https://docs.baseten.co/models/deploying-models/client#stage-2-deploying-a-draft |
|                                                                                       |
| When you are ready to publish your deployed model as a new version,                   |
| pass publish=True to the baseten.deploy() command or publish the version              |
| from the Baseten UI. To skip the draft model step in future deployments,              |
| pass publish=True to the baseten.deploy() command.                                    |
|                                                                                       |
|---------------------------------------------------------------------------------------|

Deployed model info: \n{json.dumps(model_version_json, indent=4)}{scaling_message}
"""
    )

    model_id = model_version_json["id"]
    model_version_id = model_version_json["version_id"]
    # Set the newly created model model_version_json
    # be the working model for future commands
    baseten.working_model_id = model_id

    return model_id, model_version_id


def try_patch_draft_truss(
    model_name: str,
    truss_handle: TrussHandle,
) -> TryPatchDraftTrussResponse:
    dev_model_version_info = _get_draft_model_version_info(model_name)

    if dev_model_version_info is None:
        return TryPatchDraftTrussResponse.needs_full_deploy_response(
            f"No draft model found for model {model_name}."
        )

    # We need this check because a model that has not finished building will
    # not have Truss hash and signature.
    model_deployment_status = dev_model_version_info["current_model_deployment_status"]["status"]
    if model_deployment_status == "BUILDING_MODEL":
        return TryPatchDraftTrussResponse(
            succeeded=False,
            needs_full_deploy=False,
            model_version_info=dev_model_version_info,
            error_message=f"model {model_name} is currently building, please retry after it's finished.",
        )

    truss_hash = dev_model_version_info["truss_hash"]
    if truss_hash is None:
        return TryPatchDraftTrussResponse.needs_full_deploy_response(
            f"No truss_hash found for model {model_name}."
        )

    truss_signature = dev_model_version_info["truss_signature"]
    if truss_signature is None:
        return TryPatchDraftTrussResponse.needs_full_deploy_response(
            f"No truss_signature found for model {model_name}."
        )

    TrussLocalConfigHandler.add_signature(truss_hash, truss_signature)

    patch_request = truss_handle.calc_patch(truss_hash)
    if patch_request is None:
        return TryPatchDraftTrussResponse.needs_full_deploy_response("unable to calculate patch.")

    if patch_request.prev_hash == patch_request.next_hash or len(patch_request.patch_ops) == 0:
        return TryPatchDraftTrussResponse.success_response(
            dev_model_version_info, success_message="No changes observed skipping deploy"
        )

    resp = api.patch_draft_truss(
        model_name=model_name,
        client_version=baseten.__version__,
        patch=patch_request,
    )
    if resp["succeeded"] is False:
        needs_full_deploy = True
        if "needs_full_deploy" in resp:
            needs_full_deploy = resp["needs_full_deploy"]
        return TryPatchDraftTrussResponse(
            succeeded=False,
            needs_full_deploy=needs_full_deploy,
            model_version_info=dev_model_version_info,
            error_message=resp["error"],
        )

    return TryPatchDraftTrussResponse.success_response(
        dev_model_version_info,
        success_message=f"Successfully patched draft version of model `{model_name}`.",
    )


def _get_draft_model_version_info(model_name) -> Optional[dict]:
    model = api.get_model(model_name)
    versions = model["model_version"]["oracle"]["versions"]
    for version in versions:
        if version["is_draft"] is True:
            return version
    return None


def create_model_from_s3(
    model_name,
    s3_key,
    config,
    semver_bump,
    is_trusted=False,
    external_model_version_id=None,
):
    model_version_json = api.create_model_from_truss(
        model_name=model_name,
        s3_key=s3_key,
        config=config,
        semver_bump=semver_bump,
        client_version=baseten.__version__,
        is_trusted=is_trusted,
        external_model_version_id=external_model_version_id,
    )

    logger.info(f"Created model:\n{json.dumps(model_version_json, indent=4)}")
    model_id = model_version_json["id"]
    model_version_id = model_version_json["version_id"]
    # Set the newly created model model_version_json
    # be the working model for future commands
    baseten.working_model_id = model_id

    return model_id, model_version_id


def create_model_version_from_s3(
    model_id,
    s3_key,
    config,
    semver_bump,
    is_trusted=False,
    external_model_version_id=None,
):
    model_version_json = api.create_model_version_from_truss(
        model_id=model_id,
        s3_key=s3_key,
        config=config,
        semver_bump=semver_bump,
        client_version=baseten.__version__,
        is_trusted=is_trusted,
        external_model_version_id=external_model_version_id,
    )
    model_version_id = model_version_json["id"]
    return model_version_id


def _generate_model_deploy_logs(model_id):
    logger.info(f"{Fore.BLUE} Deploying model version.")
    model_version_web_url = f"{settings.get_server_url()}/models/{model_id}"
    logger.info("🏁 The model is being built and deployed right now 🏁")
    visit_message = (
        f"|  Visit {Fore.BLUE}{model_version_web_url}{Style.RESET_ALL} for deployment status  |"
    )
    visit_message_len = len(visit_message) - len(Fore.BLUE) - len(Style.RESET_ALL)
    logger.info("".join(["-" for _ in range(visit_message_len)]))
    logger.info(visit_message)
    logger.info("".join(["-" for _ in range(visit_message_len)]))
    _generate_organization_credit_logs()


def _generate_organization_credit_logs():
    credits = api.get_organization_credits()
    credit_granted_cents = credits["monetary_credit_granted"]
    if credit_granted_cents == 0:
        # If the organization has never been granted monetary credits, omit any credit-related logging
        return
    credit_balance_cents = credits["monetary_credit_balance"]
    needs_payment_method = credits["payment_method_status"] == "NEEDS_PAYMENT_METHOD"
    credit_balance_str = (
        "{:.2f}".format(round(credit_balance_cents / 100, 2))
        if credit_balance_cents != 0
        else str(credit_balance_cents)
    )
    no_credit_remaining = (
        credit_balance_cents <= 0 and credit_granted_cents != 0 and needs_payment_method
    )
    emoji_prefix = "💸"
    if credit_granted_cents == credit_balance_cents:
        emoji_prefix = "💰"
    elif no_credit_remaining:
        emoji_prefix = "🫙"
    suffix = (
        " Any previously-deployed model versions have been deactivated."
        if no_credit_remaining
        else ""
    )
    if needs_payment_method or credit_balance_cents > 0:
        logger.info(
            f"{emoji_prefix} Your workspace has ${credit_balance_str} of model resource credit remaining.{suffix}"
        )
    if needs_payment_method:
        billing_url = f"{Fore.BLUE}{settings.get_server_url()}/settings/billing{Style.RESET_ALL}"
        add_payment_method_str = (
            f"💳 Add a payment method to reactivate your models and deploy new versions: {billing_url}"
            if no_credit_remaining
            else f"💳 Add a payment method to keep your models running once credit runs out: {billing_url}"
        )
        logger.info(add_payment_method_str)


def _archive_truss(b10_truss: TrussHandle):
    try:
        truss_dir = b10_truss._spec.truss_dir
        temp_file = create_tar_with_progress_bar(truss_dir)
    except PermissionError:
        # Windows bug with Tempfile causes PermissionErrors
        temp_file = create_tar_with_progress_bar(truss_dir, delete=False)
    temp_file.file.seek(0)
    return temp_file


def build_and_deploy_truss(
    model: Any,
    model_name: Optional[str] = None,
    version_bump=Semver.MINOR.value,
    is_trusted=False,
    publish=False,
    external_model_version_id=None,
) -> BasetenDeployedModel:
    if isinstance(model, TrussHandle):
        return deploy_truss(
            model, model_name, version_bump, is_trusted, publish, external_model_version_id
        )
    truss = build_truss(model)
    return deploy_truss(
        truss, model_name, version_bump, is_trusted, publish, external_model_version_id
    )


def pull_model(model_version_id: str, directory: str = "."):
    deployed_model = BasetenDeployedModel(model_version_id=model_version_id)
    return deployed_model.pull(directory)
