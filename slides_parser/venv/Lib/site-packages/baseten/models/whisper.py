from baseten.baseten_deployed_model import BasetenDeployedModel
from baseten.common import api, settings
from baseten.models.foundational_model import FoundationalModel
from baseten.models.util import (
    get_or_create_pretrained_model,
    requests_error_handling,
    upload_file_to_s3,
)


class Whisper(FoundationalModel):
    """
    Whisper is a state-of-the-art transcription model from OpenAI.
    It is trained on 680,000 hours of multilingual and multitask supervised data.

    **Examples:**

    Invocation with a URL

    ```python
    from baseten.models import Whisper
    model = Whisper()
    transcript = model("https://baseten.s3.amazonaws.com/whisper/whisper_test.wav")
    ```

    Invocation with a local file

    ```python
    from baseten.models import Whisper
    model = Whisper()
    transcript = model("whisper_test.wav")
    ```
    """

    def __init__(self):
        self._model = self._set_user_model()

    def status(self) -> str:
        return self._model.status

    def id(self) -> str:
        return self._model.id

    def _set_user_model(self) -> BasetenDeployedModel:
        """Creates internal BasetenDeployedModel object that points to users
        deployed Whisper model. If the user does not have a deployed model,
        we will create one for them.
        """
        model_version = get_or_create_pretrained_model("Whisper")
        return BasetenDeployedModel(
            model_id=model_version["model_id"],
            truss_spec_version=model_version["truss_spec_version"],
        )

    def __call__(self, path: str, **kwargs) -> dict:
        """Generate text from an audio file. Supports local file paths or URLs.

        Args:
            path (str): Path to audio file. Can be a local file path or a URL.

        Returns:
            transcript: Dictionary containing 3 keys: `"language"`, `"segments"`, and `"text"`.

                - `"language"` is a str representing the language of the audio file.
                - `"segments"` is a list of dicts representing segments of audio.
                Each contains `"start"` and `"end"` keys corresponding to the start and end
                times of the segment in seconds. Each also contains a `"text"` key which is
                the text generated for that segment.
                - `"text"` is a string containing the complete text generated for the audio file.
        """
        is_url = path.startswith("http")
        if not is_url:
            path = upload_file_to_s3(path)

        request_body = {
            "url": path,
        }

        # TODO: Whisper model needs to have updated spec that returns status
        with requests_error_handling():
            server_response = self._model.predict(request_body)

        return server_response

    @staticmethod
    def url():
        """
        Use this static method to get a URL to the Stable Diffusion model page
        in your Blueprint project, which contains information about the model.

        **Example:**

        ```python
        from baseten.models import Whisper

        Whisper.url()
        ```
        """
        try:
            blueprint_project_id = api.get_blueprint_projects()[0]["id"]
            return (
                f"{settings.get_server_url()}/blueprint/projects/"
                f"{blueprint_project_id}?est=community-model--whisper"
            )
        except IndexError:
            raise ValueError("User does not have any blueprint projects.")
