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
"""Tests for the Ollama HTTP client (no network required)."""

import io
import json
import urllib.error
from unittest import mock

from absl.testing import absltest

from meltingpot.utils.policies import ollama_client


class PostJsonTest(absltest.TestCase):

  def test_http_error_includes_body(self):
    error = urllib.error.HTTPError(
        url='http://localhost:11434/api/chat',
        code=400,
        msg='Bad Request',
        hdrs={},
        fp=io.BytesIO(b'{"error":"Failed to load image"}'))
    with mock.patch.object(urllib.request, 'urlopen', side_effect=error):
      with self.assertRaises(ollama_client.OllamaError) as context:
        ollama_client._post_json('http://localhost:11434/api/chat', {}, 1.)
    self.assertIn('HTTP 400', str(context.exception))
    self.assertIn('Failed to load image', str(context.exception))


class GetCapabilitiesTest(absltest.TestCase):

  def test_reads_capabilities(self):
    with mock.patch.object(
        ollama_client, '_post_json',
        return_value={'capabilities': ['decision', 'tools']}) as post:
      capabilities = ollama_client.get_capabilities('tev1:4b')
    self.assertEqual(capabilities, ('decision', 'tools'))
    post.assert_called_once_with(
        'http://localhost:11434/api/show', {'model': 'tev1:4b'}, 10.)

  def test_missing_capabilities_is_empty(self):
    with mock.patch.object(ollama_client, '_post_json', return_value={}):
      self.assertEqual(ollama_client.get_capabilities('x'), ())


class SystemOneTest(absltest.TestCase):

  def test_posts_decision_request(self):
    reply = {'answers': {'action': {'type': 'choice', 'choice': '1'}}}
    questions = {
        'action': {
            'type': 'choice',
            'instructions': 'pick one',
            'criteria': {'0': 'noop', '1': 'forward'},
        },
    }
    with mock.patch.object(
        ollama_client, '_post_json', return_value=reply) as post:
      result = ollama_client.system_one(
          'tev1:4b', 'state text', questions, keep_alive='5m')
    self.assertIs(result, reply)
    post.assert_called_once_with(
        'http://localhost:11434/v1/systemone',
        {
            'model': 'tev1:4b',
            'state': 'state text',
            'questions': questions,
            'keep_alive': '5m',
        },
        ollama_client.DEFAULT_TIMEOUT)

  def test_reply_without_answers_raises(self):
    with mock.patch.object(
        ollama_client, '_post_json', return_value={'error': 'nope'}):
      with self.assertRaises(ollama_client.OllamaError):
        ollama_client.system_one('tev1:4b', 'state', {})


class ChatTest(absltest.TestCase):

  def test_returns_message_content(self):
    reply = {'message': {'role': 'assistant', 'content': 'ACTION: 2'}}
    messages = [{'role': 'user', 'content': 'hello'}]
    with mock.patch.object(
        ollama_client, '_post_json', return_value=reply) as post:
      content = ollama_client.chat(messages, 'llama3.2:3b')
    self.assertEqual(content, 'ACTION: 2')
    url, payload, _ = post.call_args.args
    self.assertEqual(url, 'http://localhost:11434/api/chat')
    self.assertEqual(payload['model'], 'llama3.2:3b')
    self.assertEqual(payload['messages'], messages)
    self.assertFalse(payload['stream'])

  def test_expected_payload_options(self):
    with mock.patch.object(
        ollama_client, '_post_json',
        return_value={'message': {'content': 'ok'}}) as post:
      ollama_client.chat(
          [{'role': 'user', 'content': 'hi'}], 'm',
          temperature=0.1, seed=7, num_predict=32)
    payload = post.call_args.args[1]
    self.assertEqual(payload['options'],
                     {'temperature': 0.1, 'num_predict': 32, 'seed': 7})


class ListModelsTest(absltest.TestCase):

  def test_reads_tags(self):
    class _Response:

      def __enter__(self):
        return self

      def __exit__(self, *args):
        return False

      def read(self):
        return json.dumps({'models': [{'name': 'a'}, {'name': 'b'}]}).encode()

    with mock.patch.object(
        urllib.request, 'urlopen', return_value=_Response()) as urlopen:
      models = ollama_client.list_models()
    self.assertEqual(models, ('a', 'b'))
    request = urlopen.call_args.args[0]
    self.assertTrue(request.full_url.endswith('/api/tags'))


if __name__ == '__main__':
  absltest.main()
