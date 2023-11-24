from typing import Optional

from baseten.baseten_deployed_model import BasetenDeployedModel
from baseten.models.foundational_model import FoundationalModel
from baseten.models.util import requests_error_handling


class Llama(FoundationalModel):
    """
    LLaMA is a text-to-text generation model from Meta.

    !!! note
        This `Llama` model object is only available as the interface for a fine-tuned model.

    **Examples:**

    ```python
    model = Llama(model_id="llama-1234")
    completion = model("What is the meaning of life?", max_length=256)
    ```
    """

    def __init__(self, model_id: Optional[str] = None):
        """
        Args:
            model_id: The ID for a deployed model created from a LLaMA fine-tuning run.
        """

        self._id = model_id
        self._is_user_model = self._id is not None
        self._model = self._set_user_model()

    def status(self) -> str:
        return self._model.status

    def id(self) -> str:
        return self._model.id

    def _set_user_model(self) -> BasetenDeployedModel:
        """Creates internal BasetenDeployedModel object that points to users
        deployed LLaMA model. If the user does not have a deployed model,
        we will create one for them.
        """
        if self._is_user_model:
            return BasetenDeployedModel(model_id=self._id)
        else:
            raise ValueError("No model id provided")

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

        with requests_error_handling():
            server_response = self._model.predict(request_body)

        if server_response["status"] == "error":
            raise ValueError(server_response["message"])
        return server_response["data"]

    @staticmethod
    def url():
        raise ValueError("Llama is only supported as a user-finetuned model")
