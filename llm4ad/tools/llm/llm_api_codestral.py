import traceback
from mistralai import Mistral
import requests
import traceback
import time
from ...base import LLM

class MistralApi(LLM):
    def __init__(self, keys: str, model: str = "codestral-latest", timeout=60, max_retries=3, **kwargs):
        super().__init__(**kwargs)
        self._client = Mistral(api_key=keys)
        self.model_name = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._api_key = keys
        self._max_retries = max_retries

    def draw_sample(self, prompt: str, suffix: str = "", *args, **kwargs) -> str:
        temperature = kwargs.get("temperature", 1)
        retries = 0

        while retries <= self._max_retries:
            try:
                print(f"[INFO] Calling Mistral model: {self.model_name} with key {self._api_key[:8]}...")

                response = self._client.chat.complete(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )

                return response.choices[0].message.content

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                print(f"[WARN] Network issue or timeout encountered. (Attempt {retries+1}/{self._max_retries})")
                if retries < self._max_retries:
                    time.sleep(2)  # short backoff before retry
                    retries += 1
                    continue
                return "API_TIMEOUT"

            except Exception:
                print(f"[ERROR] Mistral API call failed:\n{traceback.format_exc()}")
                return "API_FAILED"
