from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from baseten.common.api import get_training_dataset
from baseten.common.files import DatasetTrainingType, upload_dataset


class DatasetIdentifier(ABC):
    """Base class for all the possible ways to indentify a dataset.
    You must provide a DatasetIdentifier to a FinetuneConfig to supply input data for the FinetuningRun.
    """

    @abstractmethod
    def resolve_dataset_url(self, training_type: DatasetTrainingType) -> str:
        """
        Method that returns a URL from which the training job can download
        a training dataset.
        """


class Dataset(DatasetIdentifier):
    """
    A Dataset hosted on Baseten

    **Example:**

    ```python
    from baseten.training import Dataset
    dataset = Dataset("DATASET_ID")
    ```
    """

    def __init__(self, dataset_id: str):
        """
        Args:
            dataset_id: The ID of the dataset hosted on Baseten
        """
        self.dataset_id = dataset_id

    def resolve_dataset_url(self, training_type: DatasetTrainingType) -> str:
        training_dataset = get_training_dataset(dataset_id=self.dataset_id)
        return training_dataset["presigned_s3_get_url"]


class LocalPath(DatasetIdentifier):
    """
    A local dataset to be uploaded

    **Example:**

    ```python
    from baseten.training import LocalPath
    dataset = LocalPath("./my-dataset")
    ```

    """

    def __init__(self, path: str, dataset_name: Optional[str] = None):
        """
        Args:
            path: The absolute or relative path to the dataset directory on your local machine
            dataset_name: The name to assign the dataset once it is uploaded to Baseten.
                If none, the uploaded directory's name will be used instead.
        """
        self.path: Path = Path(path)
        self.dataset_name: Optional[str] = dataset_name

    def resolve_dataset_url(self, training_type: DatasetTrainingType) -> str:
        dataset_id = upload_dataset(
            path=self.path, training_type=training_type, name=self.dataset_name
        )
        training_dataset = get_training_dataset(dataset_id=dataset_id)

        return training_dataset["presigned_s3_get_url"]


class PublicUrl(DatasetIdentifier):
    """
    A dataset hosted at a publicly accessible url

    **Example:**

    ```python
    from baseten.training import PublicUrl
    dataset = PublicUrl("https://cdn.baseten.co/docs/production/DreamboothSampleDataset.zip")
    ```
    """

    def __init__(self, url: str):
        """
        Args:
            url: The URL of a zip file to use as a dataset. This URL must be publicly accessible (unauthenticated)
        """
        self.url = url

    def resolve_dataset_url(self, training_type: DatasetTrainingType) -> str:
        return self.url
