# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal client for a local Ollama server.

Melting Pot policies normally load pretrained neural models from disk. This
module instead talks to a locally running Ollama daemon (https://ollama.com)
so that large language models (LLMs) can act as simulation agents. Only the
Python standard library is used so that no extra dependency is required.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

DEFAULT_BASE_URL = 'http://localhost:11434'
DEFAULT_TIMEOUT = 120


class OllamaError(RuntimeError):
  """Raised when the Ollama server cannot be reached or returns an error."""


def list_models(base_url: str = DEFAULT_BASE_URL,
                timeout: float = 10.) -> tuple[str, ...]:
  """Returns the model names available on the Ollama server."""
  request = urllib.request.Request(
      url=f'{base_url.rstrip("/")}/api/tags',
      headers={'Accept': 'application/json'})
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      payload = json.loads(response.read().decode('utf-8'))
  except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
    raise OllamaError(
        f'Could not reach the Ollama server at {base_url}: {error}') from error
  return tuple(model.get('name', '') for model in payload.get('models', ()))


def chat(messages: Sequence[Mapping[str, str]],
         model: str,
         *,
         base_url: str = DEFAULT_BASE_URL,
         temperature: float = 0.2,
         seed: int | None = None,
         num_predict: int = 128,
         timeout: float = DEFAULT_TIMEOUT,
         keep_alive: str = '30m') -> str:
  """Sends a chat request to Ollama and returns the reply text.

  Args:
    messages: Chat messages, e.g. [{"role": "user", "content": "..."}]. A
      message may additionally carry an "images" entry, a list of base64
      encoded images, for vision-capable models.
    model: The Ollama model name, e.g. "llama3.2:3b".
    base_url: Base URL of the Ollama server.
    temperature: Sampling temperature.
    seed: Sampling seed, or None for non-deterministic sampling.
    num_predict: Maximum number of tokens to generate.
    timeout: Request timeout in seconds (a single LLM call can take a while).
    keep_alive: How long the model stays loaded in Ollama between calls.

  Returns:
    The text content of the assistant's reply.

  Raises:
    OllamaError: if the request fails or the reply cannot be parsed.
  """
  payload = {
      'model': model,
      'messages': [dict(message) for message in messages],
      'stream': False,
      'keep_alive': keep_alive,
      'options': {
          'temperature': temperature,
          'num_predict': num_predict,
      },
  }
  if seed is not None:
    payload['options']['seed'] = seed

  request = urllib.request.Request(
      url=f'{base_url.rstrip("/")}/api/chat',
      data=json.dumps(payload).encode('utf-8'),
      headers={'Content-Type': 'application/json'})
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      data = json.loads(response.read().decode('utf-8'))
  except urllib.error.HTTPError as error:
    detail = error.read().decode('utf-8', errors='replace')[:500]
    raise OllamaError(
        f'Ollama chat request failed on {model}: HTTP {error.code}: {detail}'
    ) from error
  except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
    raise OllamaError(f'Ollama chat request failed on {model}: {error}') \
        from error

  message = data.get('message') or {}
  if not isinstance(message, Mapping):
    raise OllamaError(f'Ollama reply from {model} has an unexpected format.')
  return str(message.get('content', ''))
