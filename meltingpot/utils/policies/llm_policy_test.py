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
"""Tests for meltingpot.utils.policies.llm_policy (no network required)."""

import base64
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
import numpy as np

from meltingpot.utils.policies import llm_policy
from meltingpot.utils.policies import ollama_client

_ACTION_SET = (
    {'move': 0, 'turn': 0, 'fireZap': 0},
    {'move': 1, 'turn': 0, 'fireZap': 0},
    {'move': 0, 'turn': 1, 'fireZap': 0},
    {'move': 0, 'turn': 0, 'fireZap': 1},
)
_ACTION_NAMES = (
    'do nothing (standing still)',
    'step forward',
    'turn right',
    'fire a zap (attack the player ahead)',
)


# A reply mixing a reasoning section with the requested ACTION line.
PROCESSED_THINK_REPLY = ('<think>suspicious'
                         '</think> 3 verified')


def _first_timestep():
  return dm_env.TimeStep(
      step_type=dm_env.StepType.FIRST,
      reward=np.float64(0.),
      discount=None,
      observation={'RGB': np.zeros((88, 88, 3), dtype=np.uint8)})


def _mid_timestep(reward=0.25, rgb=None):
  return dm_env.TimeStep(
      step_type=dm_env.StepType.MID,
      reward=np.float64(reward),
      discount=np.float64(0.999),
      observation={
          'RGB': (np.zeros((88, 88, 3), dtype=np.uint8) if rgb is None
                  else rgb),
      })


def _policy(chat_fn, **kwargs):
  defaults = dict(
      model='test-model',
      action_names=_ACTION_NAMES,
      persona='tester',
      substrate_name='commons_harvest__open',
      chat_fn=chat_fn)
  defaults.update(kwargs)
  return llm_policy.OllamaPolicy(**defaults)


class DescribeActionTest(parameterized.TestCase):

  def test_describe_action_set(self):
    self.assertEqual(
        llm_policy.describe_action_set(_ACTION_SET), _ACTION_NAMES)

  def test_describe_action_full(self):
    self.assertEqual(
        llm_policy.describe_action(
            {'move': 1, 'turn': 1, 'fireZap': 1}),
        'step forward and turn right and fire a zap (attack the player ahead)'
    )

  def test_describe_action_unknown_key(self):
    self.assertEqual(
        llm_policy.describe_action({'move': 2, 'weirdKey': 3}),
        'sidestep right and weirdKey')

  def test_describe_action_noop(self):
    self.assertEqual(
        llm_policy.describe_action({'move': 0, 'turn': 0}), _ACTION_NAMES[0])


class RenderAsciiMapTest(parameterized.TestCase):

  def test_shape(self):
    rgb = np.full((88, 88, 3), 128, dtype=np.uint8)
    the_map = llm_policy.render_ascii_map(rgb, cells=20)
    rows = the_map.splitlines()
    self.assertLen(rows, 20)
    for row in rows:
      self.assertLen(row, 20)

  def test_dominant_colour_wins(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[:88, :88, :] = (0, 0, 0)
    rgb[0:4, 0:4, :] = (220, 60, 60)  # exactly one block is red
    the_map = llm_policy.render_ascii_map(rgb, cells=22)
    self.assertEqual(the_map.splitlines()[0][0], 'R')
    self.assertEqual(the_map.splitlines()[1][0], ' ')

  def test_colours_resolve_to_palette(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[:44, :, :] = (220, 60, 60)  # red
    the_map = llm_policy.render_ascii_map(rgb, cells=22)
    # Black maps to the void character, red maps to 'R'.
    self.assertEqual(set(the_map.replace('\n', '')), {'R', ' '})

  def test_legend_excludes_implicit_chars(self):
    legend = llm_policy._bitmap_legend()
    self.assertIn("'R'=red", legend)
    self.assertNotIn("'.'", legend)


class ParseActionTest(parameterized.TestCase):

  def test_standard_line(self):
    self.assertEqual(llm_policy._parse_action('ACTION: 2', 4), 2)

  def test_lowercase_and_explanation(self):
    self.assertEqual(
        llm_policy._parse_action('action: 1 because apples are near', 4), 1)

  def test_thinking_removed(self):
    self.assertEqual(
        llm_policy._parse_action(PROCESSED_THINK_REPLY, 4), 3)

  def test_bare_number(self):
    self.assertEqual(llm_policy._parse_action('  3  ', 4), 3)

  def test_out_of_range_is_none(self):
    self.assertIsNone(llm_policy._parse_action('ACTION: 9', 4))

  def test_garbage_is_none(self):
    self.assertIsNone(llm_policy._parse_action('I would step forward.', 4))

  def test_no_more_than_first_match(self):
    self.assertEqual(
        llm_policy._parse_action('ACTION: 1 (do not pick 2 or 3)', 4), 1)


class OllamaPolicyStepTest(parameterized.TestCase):

  def test_first_step_uses_reply_action(self):
    chat_fn = mock.MagicMock(return_value='ACTION: 2 turn right')
    agent = _policy(chat_fn)
    action, state = agent.step(_first_timestep(), agent.initial_state())
    self.assertEqual(action, 2)
    self.assertEqual(state.memory[-1], (2, 'turn right', 0.0))
    self.assertIsNone(state.decision['reasoning'] or None)

  def test_memory_resets_on_new_episode(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    state = agent.initial_state()
    state = agent.step(_first_timestep(), state)[1]
    state = agent.step(_mid_timestep(), state)[1]
    self.assertLen(state.memory, 2)
    # A FIRST timestep must wipe memory.
    state = agent.step(_first_timestep(), state)[1]
    self.assertEqual(state.memory, ((1, 'step forward', 0.0),))

  def test_reward_is_attached_to_the_action_that_earned_it(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    state = agent.step(_first_timestep(), agent.initial_state())[1]
    state = agent.step(_mid_timestep(reward=0.5), state)[1]
    self.assertEqual(state.memory[0], (1, 'step forward', 0.5))
    self.assertEqual(state.memory[-1], (1, 'step forward', 0.0))

  def test_vision_request_contains_a_png(self):
    chat_fn = mock.MagicMock(return_value='ACTION: 1')
    agent = _policy(chat_fn, vision=True)
    agent.step(_mid_timestep(), agent.initial_state())
    sent_messages = chat_fn.call_args.args[0]
    images = sent_messages[-1]['images']
    self.assertLen(images, 1)
    png = base64.b64decode(images[0])
    self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))

  def test_noop_fallback_on_gibberish(self):
    agent = _policy(mock.MagicMock(return_value='Let me think about that.'))
    action, state = agent.step(_mid_timestep(reward=1.0),
                               agent.initial_state())
    self.assertEqual(action, 0)
    self.assertIn('NOOP fallback', state.decision['note'])

  def test_service_error_falls_back_to_noop(self):
    def failing_chat(*args, **kwargs):
      del args, kwargs
      raise ollama_client.OllamaError('server unreachable')
    agent = _policy(failing_chat)
    action, state = agent.step(_mid_timestep(), agent.initial_state())
    self.assertEqual(action, 0)
    self.assertIn('ollama_error', state.decision['note'])

  def test_retry_after_bad_format(self):
    replies = iter(['choose the best move sir', 'ACTION: 3'])
    chat_fn = mock.MagicMock(side_effect=lambda *a, **k: next(replies))
    agent = _policy(chat_fn)
    action, _ = agent.step(_mid_timestep(), agent.initial_state())
    self.assertEqual(action, 3)
    self.assertEqual(chat_fn.call_count, 2)

  def test_prompt_contains_map_and_actions(self):
    chat_fn = mock.MagicMock(return_value='ACTION: 0')
    agent = _policy(chat_fn, memory_length=2)
    prompt, images = agent._build_user_prompt(
        _mid_timestep(), memory=((1, 'step forward', -0.5),))
    self.assertEqual(images, [])
    self.assertIn('commons_harvest__open', agent._system_prompt)
    self.assertIn('action', agent._system_prompt.lower())
    self.assertIn('Map legend', prompt)
    self.assertIn('step forward', prompt)
    self.assertIn('[Recent history]', prompt)
    self.assertIn('-0.50', prompt)


if __name__ == '__main__':
  absltest.main()
