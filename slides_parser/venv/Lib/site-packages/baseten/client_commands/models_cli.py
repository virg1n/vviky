import click

from baseten.common import api
from baseten.common.model_deployer import pull_model


@click.group()
def models():
    """Manage your models hosted on Baseten"""
    pass


@models.command()
def list():
    """Lists all user models"""
    pretty_print_models(api.models()["models"])


@models.command()
@click.argument("model_name")
def get(model_name):
    """Gets model information given name"""
    pretty_print_models([api.get_model(model_name)["model_version"]["oracle"]])


@models.command()
@click.option("--model_version_id", prompt="Model Version Id")
@click.option("--directory", prompt="Target Directory")
def pull(model_version_id, directory=None):
    """Pulls models given model_version id"""
    pull_model(model_version_id, directory)


def pretty_print_models(models):
    for model in models:
        semver_to_version_id = []
        for model_version in model["versions"]:
            semver_to_version_id.append({f"Version {model_version['semver']}": model_version["id"]})
        click.echo(f"Name: {model['name']} | Versions: {semver_to_version_id}")
