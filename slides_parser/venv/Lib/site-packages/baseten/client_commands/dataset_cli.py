from pathlib import Path
from typing import Optional

import click

from baseten.common.files import DatasetTrainingType, upload_dataset


@click.group(name="dataset")
def dataset_cli():
    """Manage datasets stored on Blueprint"""
    pass


@dataset_cli.command()
@click.argument("data_dir", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option(
    "--name",
    "-n",
    required=False,
    type=str,
    help="Name given to Dataset on Blueprint. If none, the uploaded directory's name will be used instead.",
)
@click.option(
    "--training-type",
    "-t",
    required=True,
    type=click.Choice([t.value for t in DatasetTrainingType], case_sensitive=False),
    help="Types of FinetuningRun that can use this Dataset",
)
def upload(data_dir: Path, name: Optional[str], training_type: str):
    """Upload a new dataset to Baseten from DATA_DIR

    DATA_DIR is the path to directory with relevant files.
    """
    dataset_id = upload_dataset(
        path=Path(data_dir), training_type=DatasetTrainingType[training_type.upper()], name=name
    )
    click.echo()
    click.echo(f"Dataset ID:\n{dataset_id}")
