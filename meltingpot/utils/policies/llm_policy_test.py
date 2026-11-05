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
      backend='chat',
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


class BackendSelectionTest(parameterized.TestCase):

  def test_auto_detects_decision_models(self):
    agent = _policy(
        mock.MagicMock(), backend='auto',
        capabilities_fn=lambda *a, **k: ('decision',))
    self.assertEqual(agent.backend, 'decision')

  def test_auto_detects_chat_models(self):
    agent = _policy(
        mock.MagicMock(), backend='auto',
        capabilities_fn=lambda *a, **k: ('completion', 'tools'))
    self.assertEqual(agent.backend, 'chat')

  def test_auto_falls_back_to_chat_if_server_unreachable(self):
    def failing_capabilities(*args, **kwargs):
      del args, kwargs
      raise ollama_client.OllamaError('server unreachable')
    agent = _policy(mock.MagicMock(), backend='auto',
                    capabilities_fn=failing_capabilities)
    self.assertEqual(agent.backend, 'chat')

  def test_explicit_backend_skips_detection(self):
    capabilities = mock.MagicMock()
    agent = _policy(mock.MagicMock(), backend='decision',
                    capabilities_fn=capabilities)
    self.assertEqual(agent.backend, 'decision')
    capabilities.assert_not_called()

  def test_invalid_backend_is_rejected(self):
    with self.assertRaises(ValueError):
      _policy(mock.MagicMock(), backend='magic')


def _decision_reply(choice='2', confidence=0.9):
  return {
      'model': 'test-model',
      'answers': {
          'action': {
              'type': 'choice',
              'choice': choice,
              'probabilities': {'0': 0.02, '1': 0.05, '2': 0.9, '3': 0.03},
              'confidence': confidence,
          },
      },
      'usage': {'input_tokens': 100, 'output_tokens': 0},
  }


class DecisionBackendTest(parameterized.TestCase):

  def test_choice_is_used_as_the_action(self):
    decision_fn = mock.MagicMock(return_value=_decision_reply('2'))
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=decision_fn)
    action, state = agent.step(_first_timestep(), agent.initial_state())
    self.assertEqual(action, 2)
    self.assertEqual(state.memory[-1], (2, 'turn right', 0.0))
    self.assertEqual(state.decision['backend'], 'decision')
    self.assertEqual(state.decision['choice'], '2')
    self.assertAlmostEqual(state.decision['confidence'], 0.9)
    self.assertIsNone(state.decision['note'])

  def test_request_uses_choice_question(self):
    decision_fn = mock.MagicMock(return_value=_decision_reply('1'))
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=decision_fn)
    agent.step(_mid_timestep(), agent.initial_state())
    kwargs = decision_fn.call_args.kwargs
    self.assertIn('Your local view', kwargs['state'])
    self.assertIn('Map legend', kwargs['state'])
    question = kwargs['questions']['action']
    self.assertEqual(question['type'], 'choice')
    self.assertEqual(question['criteria']['3'],
                     'fire a zap (attack the player ahead)')
    self.assertIn('maximises the reward', question['instructions'])

  def test_invalid_choice_falls_back_to_noop(self):
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=mock.MagicMock(
                        return_value=_decision_reply('99')))
    action, state = agent.step(_mid_timestep(), agent.initial_state())
    self.assertEqual(action, 0)
    self.assertIn('NOOP fallback', state.decision['note'])

  def test_missing_choice_falls_back_to_noop(self):
    reply = {'answers': {'action': {'type': 'choice'}}}
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=mock.MagicMock(return_value=reply))
    action, _ = agent.step(_mid_timestep(), agent.initial_state())
    self.assertEqual(action, 0)

  def test_decision_error_falls_back_to_noop(self):
    def failing_decision(*args, **kwargs):
      del args, kwargs
      raise ollama_client.OllamaError('endpoint missing')
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=failing_decision)
    action, state = agent.step(_mid_timestep(), agent.initial_state())
    self.assertEqual(action, 0)
    self.assertIn('ollama_error', state.decision['note'])

  def test_decision_backend_does_not_send_images(self):
    decision_fn = mock.MagicMock(return_value=_decision_reply('0'))
    agent = _policy(mock.MagicMock(), backend='decision', vision=True,
                    decision_fn=decision_fn)
    agent.step(_mid_timestep(), agent.initial_state())
    self.assertNotIn('images', str(decision_fn.call_args.kwargs))


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


class TargetSummaryTest(parameterized.TestCase):

  def test_finds_apples_and_reports_direction(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[8:16, 8:16, :] = (220, 60, 60)  # red apples, upper-left
    summary = llm_policy._summarise_targets(rgb, 'commons_harvest__open')
    self.assertIn('red apples', summary)
    self.assertIn('ahead', summary)  # upper part of the view is ahead
    self.assertIn('left', summary)   # and it is on the left
    self.assertIn('cells away', summary)

  def test_recommendation_includes_action_number(self):
    names = ('do nothing (standing still)', 'step forward', 'step backward',
             'sidestep left', 'sidestep right', 'turn left', 'turn right',
             'fire a zap (attack the player ahead)')
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[8:16, 8:16, :] = (220, 60, 60)  # ahead-left
    summary = llm_policy._summarise_targets(
        rgb, 'commons_harvest__open', action_names=names)
    self.assertIn('ACTION 5', summary)  # turn left

    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[8:16, 36:52, :] = (220, 60, 60)  # ahead
    summary = llm_policy._summarise_targets(
        rgb, 'commons_harvest__open', action_names=names)
    self.assertIn('ACTION 1', summary)  # step forward

  def test_reports_missing_targets(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    summary = llm_policy._summarise_targets(rgb, 'commons_harvest__open')
    self.assertIn('not visible', summary)

  def test_no_visible_target_recommends_scanning(self):
    names = ('do nothing (standing still)', 'step forward', 'step backward',
             'sidestep left', 'sidestep right', 'turn left', 'turn right',
             'fire a zap (attack the player ahead)')
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    summary = llm_policy._summarise_targets(
        rgb, 'commons_harvest__open', action_names=names)
    self.assertIn('not visible', summary)
    self.assertIn('ACTION 6', summary)  # turn right to scan

  def test_clean_up_recommends_cleaning_when_river_is_ahead(self):
    names = ('do nothing (standing still)', 'step forward', 'step backward',
             'sidestep left', 'sidestep right', 'turn left', 'turn right',
             'fire a zap (attack the player ahead)',
             'fire a cleaning beam (clean polluted cells)')
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[:40, :, :] = (70, 110, 220)  # blue river in the upper (ahead) half
    summary = llm_policy._summarise_targets(
        rgb, 'clean_up', action_names=names)
    self.assertIn('ACTION 8', summary)
    self.assertIn('cleaning beam', summary)

  def test_clean_up_turns_to_search_after_cleaning(self):
    names = ('do nothing (standing still)', 'step forward', 'step backward',
             'sidestep left', 'sidestep right', 'turn left', 'turn right',
             'fire a zap (attack the player ahead)',
             'fire a cleaning beam (clean polluted cells)')
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[:40, :, :] = (70, 110, 220)
    summary = llm_policy._summarise_targets(
        rgb, 'clean_up', action_names=names,
        last_action_name='fire a cleaning beam (clean polluted cells)')
    self.assertIn('You just cleaned the river', summary)
    self.assertIn('ACTION 6', summary)  # turn right to search for apples

  def test_generic_substrate_uses_notable_colours(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[10:20, 10:20, :] = (240, 220, 80)  # yellow, upper-left
    summary = llm_policy._summarise_targets(rgb, 'some_unknown_game')
    self.assertIn('yellow objects', summary)

  def test_small_vector_observations_are_formatted(self):
    timestep = _mid_timestep()
    observation = dict(timestep.observation)
    observation['POSITION'] = np.array([1, 2])
    observation['STAMINA'] = np.array([3, 4, 5])
    timestep = timestep._replace(observation=observation)
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    prompt, _ = agent._build_user_prompt(timestep, memory=())
    self.assertIn('POSITION', prompt)
    self.assertIn('STAMINA', prompt)

  def test_prompt_contains_targets_and_progress(self):
    rgb = np.zeros((88, 88, 3), dtype=np.uint8)
    rgb[8:16, 8:16, :] = (220, 60, 60)
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    prompt, _ = agent._build_user_prompt(
        _mid_timestep(rgb=rgb), memory=(), total_reward=2.5)
    self.assertIn('[Targets near you]', prompt)
    self.assertIn('red apples', prompt)
    self.assertIn('[Progress]', prompt)
    self.assertIn('+2.50', prompt)


class ProgressTextTest(parameterized.TestCase):

  def test_progress_reports_total_and_streak(self):
    text = llm_policy._progress_text(3.5, 0, 'en')
    self.assertIn('+3.50', text)
    self.assertIn('without reward: 0', text)

  def test_progress_nudges_after_two_zero_reward_steps(self):
    text = llm_policy._progress_text(0.0, 2, 'en', stalled=True)
    self.assertIn('Change strategy', text)

  def test_progress_keeps_going_while_moving(self):
    text = llm_policy._progress_text(0.0, 3, 'en', stalled=False)
    self.assertIn('Follow the recommended next action', text)
    self.assertNotIn('Change strategy', text)

  def test_action_stall_detects_repeats_and_noop(self):
    self.assertFalse(llm_policy._is_action_stall(
        ((2, 'turn right', 0.0), (2, 'turn right', 0.0))))
    self.assertTrue(llm_policy._is_action_stall(
        ((2, 'turn right', 0.0), (2, 'turn right', 0.0),
         (2, 'turn right', 0.0))))
    self.assertTrue(llm_policy._is_action_stall(
        ((0, 'do nothing', 0.0), (0, 'do nothing', 0.0))))
    self.assertFalse(llm_policy._is_action_stall(
        ((1, 'step forward', 0.0), (1, 'step forward', 0.0))))
    self.assertFalse(llm_policy._is_action_stall(
        ((1, 'step forward', 0.0), (2, 'turn right', 0.0))))

  def test_zero_streak_counts_from_the_end(self):
    memory = ((1, 'step forward', 0.0), (2, 'turn right', 1.0),
              (1, 'step forward', 0.0), (0, 'do nothing', 0.0))
    self.assertEqual(llm_policy._zero_streak(memory), 2)


class ExplorationTest(parameterized.TestCase):

  def _stalled_state(self):
    return llm_policy.LLMAgentState(
        memory=((0, 'do nothing (standing still)', 0.0),
                (0, 'do nothing (standing still)', 0.0)))

  def test_stalled_noop_is_overridden_by_movement(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 0'))
    action, state = agent.step(_mid_timestep(reward=0.0), self._stalled_state())
    self.assertNotEqual(action, 0)
    self.assertEqual(action, agent._exploration_pattern[0])
    self.assertTrue(state.decision['exploration'])
    self.assertIn('stalled: exploration override', state.decision['note'])

  def test_repeated_forward_is_not_overridden(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    state = llm_policy.LLMAgentState(
        memory=((1, 'step forward', 0.0), (1, 'step forward', 0.0)))
    action, next_state = agent.step(_mid_timestep(reward=0.0), state)
    self.assertEqual(action, 1)
    self.assertFalse(next_state.decision['exploration'])

  def test_two_turns_are_allowed_for_scanning(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 2'))
    state = llm_policy.LLMAgentState(
        memory=((2, 'turn right', 0.0), (2, 'turn right', 0.0)))
    action, next_state = agent.step(_mid_timestep(reward=0.0), state)
    self.assertEqual(action, 2)
    self.assertFalse(next_state.decision['exploration'])

  def test_three_repeated_turns_are_overridden(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 2'))
    state = llm_policy.LLMAgentState(
        memory=((2, 'turn right', 0.0), (2, 'turn right', 0.0),
                (2, 'turn right', 0.0)))
    action, next_state = agent.step(_mid_timestep(reward=0.0), state)
    self.assertNotEqual(action, 2)
    self.assertTrue(next_state.decision['exploration'])

  def test_new_action_is_not_overridden(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    action, state = agent.step(_mid_timestep(reward=0.0), self._stalled_state())
    self.assertEqual(action, 1)
    self.assertFalse(state.decision['exploration'])

  def test_reward_prevents_override(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 0'))
    action, state = agent.step(_mid_timestep(reward=0.5), self._stalled_state())
    self.assertEqual(action, 0)
    self.assertFalse(state.decision['exploration'])
    self.assertEqual(state.total_reward, 0.5)

  def test_single_zero_step_does_not_trigger(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 0'))
    prev = llm_policy.LLMAgentState(
        memory=((0, 'do nothing (standing still)', 0.0),))
    action, state = agent.step(_mid_timestep(reward=0.0), prev)
    self.assertEqual(action, 0)
    self.assertFalse(state.decision['exploration'])

  def test_exploration_can_be_disabled(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 0'), exploration=False)
    action, state = agent.step(_mid_timestep(reward=0.0), self._stalled_state())
    self.assertEqual(action, 0)
    self.assertFalse(state.decision['exploration'])

  def test_decision_backend_is_also_explored(self):
    agent = _policy(mock.MagicMock(), backend='decision',
                    decision_fn=mock.MagicMock(
                        return_value=_decision_reply('0')))
    action, state = agent.step(_mid_timestep(reward=0.0), self._stalled_state())
    self.assertNotEqual(action, 0)
    self.assertTrue(state.decision['exploration'])
    self.assertEqual(state.decision['choice'], '0')  # model choice is logged

  def test_total_reward_accumulates_across_steps(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    state = agent.initial_state()
    _, state = agent.step(_first_timestep(), state)
    _, state = agent.step(_mid_timestep(reward=0.5), state)
    self.assertEqual(state.total_reward, 0.5)
    _, state = agent.step(_mid_timestep(reward=1.0), state)
    self.assertEqual(state.total_reward, 1.5)

  def test_new_episode_resets_total_reward(self):
    agent = _policy(mock.MagicMock(return_value='ACTION: 1'))
    state = llm_policy.LLMAgentState(total_reward=9.0)
    _, state = agent.step(_first_timestep(), state)
    self.assertEqual(state.total_reward, 0.0)


if __name__ == '__main__':
  absltest.main()
