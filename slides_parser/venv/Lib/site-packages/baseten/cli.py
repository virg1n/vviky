"""Console script for baseten."""

import configparser
import functools
import logging
import platform
import sys
import time

import click
import requests
from requests.exceptions import RequestException
from truss.cli import truss_cli as truss_commands

from baseten import version as b10_version
from baseten.client_commands import (
    baseten_cli,
    dataset_cli,
    finetuning_cli,
    models_cli,
    pretrained_cli,
)
from baseten.common.settings import (
    get_server_url,
    read_config,
    set_config_value,
    set_server_url,
)
from baseten.common.util import setup_logger

logger = logging.getLogger(__name__)

DEFAULT_NETWORK_ERROR_MESSAGE = (
    "Sorry, something went wrong on our end. Please try again in 5 minutes."
)
# Message to display after getting a response from the server
SIGNUP_ERROR_MESSAGE = (
    "\nIf you have already signed up, please visit "
    "https://app.baseten.co/blueprint/tokens to "
    "initiate a login."
)


def ensure_login(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = read_config()
        try:
            config.get("api", "api_key")
        except configparser.NoOptionError:
            click.echo("You must first run the `baseten login` cli command.")
            sys.exit()
        result = func(*args, **kwargs)
        return result

    return wrapper


@click.group(name="baseten", invoke_without_command=True)
@click.pass_context
@click.option(
    "-v",
    "--version",
    is_flag=True,
    show_default=False,
    default=False,
    help="Show baseten package version.",
)
def cli_group(ctx, version):
    setup_logger("baseten", logging.INFO)
    if not ctx.invoked_subcommand:
        if version:
            click.echo(b10_version())
        else:
            click.echo(ctx.get_help())


@cli_group.command()
@click.option("--server_url", prompt="Baseten server URL", help="URL to hosted Baseten solution.")
def configure(server_url):
    """Configure client to use on-prem hosted Baseten environment.

    !!! note

        The client works out of the box without any need to run this command.
        This command is only required if you have Baseten Enterprise hosted on-prem and is not
        required for most users.
    """
    if set_server_url(server_url):
        click.echo("Saved server URL.")
    else:
        click.echo("That is not a valid URL.")


@cli_group.command()
@click.option(
    "--api_key", prompt="Baseten API key", hide_input=True, help="Valid API Key from Baseten."
)
def login(api_key: str):
    """Authenticate user with Baseten using API Key."""
    set_config_value("api", "api_key", api_key)
    click.echo("Saved API key.")


@cli_group.command()
@click.option(
    "--email", prompt="Email Address", hide_input=False, help="The email to register as user with."
)
def signup(email: str):
    """Signup as new Blueprint user.

    The sign-up process:

    1. Create an API generator token
    2. Create a baseten account with the provided email address
    3. Use API generator token to poll for newly created API Key
    """
    server_url = get_server_url()

    try:
        api_generator_response = requests.post(
            server_url + "/api/create_api_generator_token",
            json={"device_name": platform.node(), "device_os": platform.platform()},
        )

        if api_generator_response.status_code != 200:
            click.echo(DEFAULT_NETWORK_ERROR_MESSAGE)
            return

        token = api_generator_response.json()["token"]

        signup_response = requests.post(
            server_url + "/api/signup_new_user",
            json={"email": email, "is_package_user": True, "token": token},
        )
    except RequestException:
        click.echo(DEFAULT_NETWORK_ERROR_MESSAGE)
        return

    if signup_response.status_code != 200:
        click.echo(DEFAULT_NETWORK_ERROR_MESSAGE)
        return

    signup_response_json = signup_response.json()

    if signup_response_json["success"]:
        click.echo("Please check your email for a confirmation")
        _poll_for_available_api_key(token)
    else:
        click.echo(signup_response_json["error"])
        click.echo(SIGNUP_ERROR_MESSAGE)


def _poll_for_available_api_key(token):
    SLEEP_BETWEEN_POLLING_SECONDS = 2
    TIMEOUT_SECONDS = 300  # Poll for up to 5 minutes

    # Divide the total timeout by the amount sleeping in loop to
    # compute total # of iterations
    for _ in range(int(TIMEOUT_SECONDS / SLEEP_BETWEEN_POLLING_SECONDS)):
        try:
            response = requests.get(get_server_url() + f"/api/api_generator_token/{token}")
        except RequestException:
            time.sleep(SLEEP_BETWEEN_POLLING_SECONDS)
            continue

        api_key = response.json().get("api_key")
        if api_key is not None:
            set_config_value("api", "api_key", api_key)

            click.echo("API Key Successfully Saved.")
            break
        else:
            time.sleep(SLEEP_BETWEEN_POLLING_SECONDS)

    else:
        click.echo(DEFAULT_NETWORK_ERROR_MESSAGE)


truss_commands.add_command(baseten_cli.deploy_from_directory)
# Add Truss command group
cli_group.add_command(truss_commands)
# Add pretrained models command group
cli_group.add_command(pretrained_cli.pretrained)
# Add models command group
cli_group.add_command(models_cli.models)
# Add dataset command group
cli_group.add_command(dataset_cli.dataset_cli)
# Add finetuning command group
cli_group.add_command(finetuning_cli.finetuning_cli)

if __name__ == "__main__":
    cli_group()
