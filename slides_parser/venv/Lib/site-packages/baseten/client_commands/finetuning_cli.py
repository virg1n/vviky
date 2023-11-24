import json
import os
from json import JSONDecodeError
from typing import Optional

import click

from baseten.training import (
    Dataset,
    DatasetIdentifier,
    DreamboothConfig,
    FinetuningRun,
    LocalPath,
    PublicUrl,
)


def _finetune_interactive(finetuning_type: str):
    raise NotImplementedError


def _validate_input_options(input_: Optional[str], input_file: Optional[str]) -> bool:

    if input_ and input_file:
        click.echo("Please provide exactly one of input|input_file")
        return False

    if input_ is None and input_file is None:
        click.echo("Please provide exactly one of input|input_file")
        return False

    if input_file is not None and (not os.path.exists(input_file) or os.path.isdir(input_file)):
        click.echo(f"{input_file} is not a valid file.")
        return False

    return True


def _parse_input_data(params: dict) -> Optional[DatasetIdentifier]:
    if "input_dataset" in params:
        raw_input_data = params["input_dataset"]
    else:
        return None

    if "public_url" in raw_input_data:
        return PublicUrl(raw_input_data["public_url"])

    if "baseten_dataset_id" in raw_input_data:
        return Dataset(raw_input_data["baseten_dataset_id"])

    if "local_path" in raw_input_data:
        if "dataset_name" in raw_input_data:
            return LocalPath(raw_input_data["local_path"], raw_input_data["dataset_name"])
        else:
            click.echo("If using local_path, please also provide a `dataset_name`.")
            return None

    return None


def _finetune_dreambooth(trained_model_name: str, params: dict) -> Optional[FinetuningRun]:
    parsed_input_dataset = _parse_input_data(params)

    if parsed_input_dataset:
        params.update({"input_dataset": parsed_input_dataset})
        config = DreamboothConfig(**params)
        return FinetuningRun.create(trained_model_name, config)
    else:
        click.echo("No valid `input_dataset` provided.")
        return None


@click.group(name="finetuning")
def finetuning_cli():
    """Run custom Finetuning jobs with Blueprint"""
    pass


@finetuning_cli.command()
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Deploy interactively using a series of guided prompts.",
)
@click.option(
    "--finetuning-type",
    type=click.Choice(["DREAMBOOTH", "CLASSIC_STABLE_DIFFUSION"]),
    required=True,
    help="Type of Finetuning run to create.",
)
@click.option("--name", required=True, help="Name for this Finetuning run.")
@click.option("--input-file", required=False, help="File pointing to the configuration.")
# TODO(sid): What is this?
@click.option("--input", "input_", required=False)
def create(interactive: bool, finetuning_type: str, name: str, input_file: str, input_: str):
    """Create and trigger new fine-tunng run

    See
    [FinetuningRun.create][baseten.training.finetuning.FinetuningRun.create]
    for more details.
    """
    if interactive:
        click.echo("Starting interactive finetuning flow.")
        _finetune_interactive(finetuning_type)
    else:
        if _validate_input_options(input_, input_file):
            # TODO: Take input or input file, parse it into a dict. Handle the input_data piece, and
            # construct a Dreambooth config object.
            if input_file:
                input_ = open(input_file, "r").read()

            try:
                parsed_json = json.loads(input_)
            except JSONDecodeError as error:
                click.echo("Could not parse JSON.")
                click.echo(str(error))
                return

            if finetuning_type == "DREAMBOOTH":
                training_run = _finetune_dreambooth(name, parsed_json)
                if training_run:
                    click.echo(f"Started training run: {training_run.id}")
                    click.echo(f"Access the training logs at {training_run.blueprint_url}")
                    click.echo(f"or use `blueprint finetunings logs --id {training_run.id}")


@finetuning_cli.command()
def list():
    """List all finetuning runs.

    See
    [FinetuningRun.list][baseten.training.finetuning.FinetuningRun.list]
    for more details.
    """
    runs = FinetuningRun.list()
    if len(runs) == 0:
        click.echo("No runs found. Kick off a run using `baseten finetuning create`.")
        return
    click.echo(f"Found {len(runs)} runs")
    for run in runs:
        click.echo(4 * " " + f"id: {run.id}, name: {run.trained_model_name}")


@finetuning_cli.command()
@click.option("--run-id", type=str, required=True, help="ID for Blueprint Finetuning Run.")
@click.option(
    "--idle-time-minutes",
    type=int,
    default=30,
    help="Number of minutes before server scales down to save costs.",
)
def deploy(run_id: str, idle_time_minutes: int):
    """Deploy model from finetuning run


    See
    [FinetuningRun.deploy][baseten.training.finetuning.FinetuningRun.deploy]
    for more details.
    """
    run = FinetuningRun(run_id)
    run.deploy(idle_time_minutes)
    click.echo(f"Access the deployed model at {run.blueprint_url}")


@finetuning_cli.command()
@click.option("--run-id", type=str, required=True, help="ID for Blueprint Finetuning Run.")
def cancel(run_id: str):
    """Cancel Finetuning run

    See
    [FinetuningRun.cancel][baseten.training.finetuning.FinetuningRun.cancel]
    for more details.
    """
    cancelled = FinetuningRun(run_id).cancel()
    if cancelled:
        click.echo(f"Successfully cancelled finetuning run {run_id}.")
    else:
        click.echo(f"Failed to cancel finetuning run {run_id}.")


@finetuning_cli.command()
@click.option("--run-id", type=str, required=True, help="ID for Blueprint Finetuning Run.")
def logs(run_id: str):
    """Stream logs for Finetuning run

    See
    [FinetuningRun.stream_logs][baseten.training.finetuning.FinetuningRun.stream_logs]
    for more details."""
    FinetuningRun(run_id).stream_logs()
