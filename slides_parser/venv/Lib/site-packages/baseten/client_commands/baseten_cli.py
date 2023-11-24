import os

import click

import baseten


@click.command()
@click.argument("target_directory", required=False)
@click.argument("model_name", required=False)
def deploy_from_directory(target_directory=None, model_name=None):
    """Deploys a Truss from a directory

    TARGET_DIRECTORY (Str): Directory of the truss

    MODEL_NAME (Str): Name of the model
    """
    if target_directory is None:
        target_directory = os.getcwd()
    truss = baseten.load_truss(target_directory)
    baseten.deploy_truss(truss, model_name=model_name)
