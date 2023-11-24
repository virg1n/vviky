from typing import Any, List, Optional

from baseten.baseten_deployed_model import BasetenDeployedModel
from baseten.common import api, settings
from baseten.models.foundational_model import FoundationalModel
from baseten.models.util import get_or_create_pretrained_model, requests_error_handling


class FlanT5(FoundationalModel):
    """
    Flan-T5 is a instruction-tuned text-to-text generation model from Google. It is built atop the generic T5 model.

    **Examples:**

    Invoking default Flan T5

    ```python
    from baseten.models import FlanT5
    model = FlanT5()
    model("The quick brown fox jumps over the lazy dog.")
    ```

    **Setting bad words**

    For production apps, you may find it useful to set a list of "bad words" which tell the model
    what words it cannot use during generation. Blueprint supports this by letting you pass that
    list during instantiation of the model.

    ```python
    model = FlanT5()
    model("The quick brown fox jumps over the lazy dog.", bad_words=["dog"])
    ```

    ```python
    model = FlanT5()
    model.bad_words = ["dog"]
    model("The quick brown fox jumps over the lazy dog.")
    ```

    ```python
    model = FlanT5()
    model.bad_words = ["dog"]
    model("The quick brown fox jumps over the lazy dog.", num_beams=4)
    ```
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Attributes:
            bad_words (list): List of words to avoid in the output.
        """
        self.bad_words: List[str] = []

        self._id = model_id
        self._is_user_model = self._id is not None
        self._model = self._set_user_model()

    def status(self) -> str:
        return self._model.status

    def id(self) -> str:
        return self._model.id

    def _set_user_model(self) -> BasetenDeployedModel:
        """Creates internal BasetenDeployedModel object that points to users
        deployed Flan-T5 model. If the user does not have a deployed model,
        we will create one for them.
        """
        if self._is_user_model:
            return BasetenDeployedModel(model_id=self._id)

        model_version = get_or_create_pretrained_model("Flan-T5 XL")
        return BasetenDeployedModel(
            model_id=model_version["model_id"],
            truss_spec_version=model_version["truss_spec_version"],
        )

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Validates attributes before setting them."""
        if __name == "bad_words":
            if not isinstance(__value, list):
                raise ValueError("bad_words must be a list")
        super().__setattr__(__name, __value)

    def __call__(self, prompt, seed=None, **kwargs) -> str:
        """Generate text from a prompt. Supports all parameters of the underlying model
        from Huggingface library. The below are _some_ of the parameters that can be passed in.
        See Huggingface Transformers `.generate()` documentation for more details.

        Args:
            prompt (str): The prompt to generate text from.
            seed (int): The random seed to use for reproducibility. Optional.
        Attributes:
            num_beams (int): Number of beams for beam search. 1 means no beam search. Optional. Default to 1.
            num_return_sequences (int): The number of independently computed returned sequences for
                each element in the batch. Default to 1.
            max_length (int): The maximum number of tokens that can be generated.
            temperature (float): The value used to module the next token probabilities. Must be
                strictly positive. Default to 1.0.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Between 1 and infinity. Default to 50.
            top_p (float): If set to float < 1, only the most probable tokens with probabilities that
                add up to top_p or higher are kept for generation. Default to 1.0.
            repetition_penalty (float): The parameter for repetition penalty. Between 1.0 and infinity.
                1.0 means no penalty. Default to 1.0.
            length_penalty (float): Exponential penalty to the sequence length. Default to 1.0.
            no_repeat_ngram_size (int): If set to int > 0, all ngrams of that size can only occur
                once. Default to 0.
            num_return_sequences (int): The number of independently computed returned sequences
                for each element in the batch. Default to 1.
            do_sample (bool): If set to False greedy decoding is used. Otherwise sampling is used.
                Defaults to True.
            early_stopping (bool): Whether to stop the beam search when at least num_beams sentences
                are finished per batch or not. Default to False.
            use_cache (bool): Whether or not the model should use the past last key/values attentions
                (if applicable to the model) to speed up decoding. Default to True.
            decoder_start_token_id (int): If an encoder-decoder model starts decoding with a different
                token than BOS, the id of that token. Default to None.
            pad_token_id (int): The id of the padding token. Default to None.
            eos_token_id (int): The id of the end of sequence token. Default to None.
            forced_bos_token_id (int): The id of the token to force as the first generated
                token after the BOS token. Default to None.
            forced_eos_token_id (int): The id of the token to force as the last generated
                token when max_length is reached. Default to None.
            remove_invalid_values (bool): Whether or not to remove possible `nan` and `inf`
                outputs of the model to prevent the generation method to crash. Default to False.

        Returns:
            generated_text (str): The generated text.
        """
        request_body = {"prompt": prompt, **kwargs}

        if self.bad_words:
            request_body["bad_words"] = self.bad_words
        if seed:
            request_body["seed"] = seed

        with requests_error_handling():
            server_response = self._model.predict(request_body)

        if server_response["status"] == "error":
            raise ValueError(server_response["message"])
        return server_response["data"]

    @staticmethod
    def url():
        """
        Use this static method to get a URL to the Stable Diffusion model page
        in your Blueprint project, which contains information about the model.

        **Example:**

        ```python
        from baseten.models import FlanT5

        FlanT5.url()
        ```
        """
        try:
            blueprint_project_id = api.get_blueprint_projects()[0]["id"]
            return (
                f"{settings.get_server_url()}/blueprint/projects/"
                f"{blueprint_project_id}?est=community-model--flan"
            )
        except IndexError:
            raise ValueError("User does not have any blueprint projects.")
