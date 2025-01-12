from typing import Any, Dict, List
from openai import AzureOpenAI

from aih_automaton.ai_models.model_base import AIModel
from aih_automaton.data_models import FileResponse
from aih_automaton.utils.resource_handler import ResourceBox


class AzureOpenAIModel(AIModel):
    def __init__(
        self,
        azure_api_key,
        azure_api_version,
        parameters: Dict[str, Any] = None,
        azure_endpoint: str = None,
    ):
        self.parameters = parameters
        self.client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )
        self.api_key = azure_api_key

    def generate_text(
        self,
        task_id: str = None,
        system_persona: str = None,
        prompt: str = None,
        messages: List[dict] = None,
        functions: List[Dict[str, Any]] = None,
        function_call: str = None,
        **kwargs,
    ):
        # task_id kept for future use
        if (
            messages is None
        ):  # and not (system_persona is None) and not(prompt is None) :
            messages = [
                {"role": "system", "content": system_persona},
                {"role": "user", "content": prompt},
            ]

        response = self.client.chat.completions.create(
            messages=messages,
            functions=functions,
            function_call=function_call,
            **self.parameters,
        )
        if functions is not None and len(functions) > 0:

            return response.choices[0].message.function_call
        # TODO ability to return multiple responses
        return response.choices[0].message.content

    def generate_image(
        self, task_id: str, prompt: str, resource_box: ResourceBox
    ) -> FileResponse:
        response = self.client.images.generate(**self.parameters, prompt=prompt)
        return resource_box.save_from_url(url=response.data[0].url, subfolder=task_id)
