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
"""LLM agent policies for Melting Pot, backed by a local Ollama server.

`OllamaPolicy` implements the standard Melting Pot `Policy` interface, so any
player slot of a substrate can be driven by a large language model. Each step
the model receives a textual description of the agent's local view (a
coarse-colour ASCII rendering of the RGB observation) plus a short memory of
its recent actions and rewards, and answers with a discrete action index.
"""

import base64
import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence

import dm_env
import numpy as np

from meltingpot.utils.policies import ollama_client
from meltingpot.utils.policies import policy
from meltingpot.utils.policies.ollama_client import DEFAULT_BASE_URL
from meltingpot.utils.policies.ollama_client import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

_PROMPT_LANGUAGES = ('en', 'ja')

# Move values follow the dmlab2d "move" convention used by the substrates.
_MOVE_DESCRIPTIONS = {
    1: 'step forward',
    2: 'sidestep right',
    3: 'step backward',
    4: 'sidestep left',
}

# Maps dmlab2d action keys (other than move/turn) to readable fragments.
_EXTRA_ACTION_NAMES = {
    'fireZap': 'fire a zap (attack the player ahead)',
    'fireClean': 'fire a cleaning beam (clean polluted cells)',
    'interact': 'interact with the space ahead',
    'fireClaim': 'fire a claim laser',
    'refineAndGift': 'refine and gift a token to the neighbour ahead',
    'consumeTokens': 'consume one of your tokens for a bonus',
    'ioAction': 'activate the IO switch ahead',
    'hold': 'hold your arms out (e.g. to play rock-paper-scissors)',
    'shove': 'shove a neighbouring player',
    'grasp': 'grasp (hold/use) the tool ahead',
    'mine': 'mine the rock ahead',
    'flail': 'flail your arms (attack neighbours)',
    'row': 'row the boat forward',
    'eat': 'eat the item you stepped on',
    'eat_apple': 'eat an apple',
    'eat_banana': 'eat a banana',
    'offer_apple': 'offer an apple to trade',
    'offer_banana': 'offer a banana to trade',
    'offer_cancel': 'cancel the current trade offer',
    'pickup': 'pick up the item ahead',
}

# (character, colour name, RGB triple) used for the ASCII world map.
_MAP_PALETTE = (
    (' ', 'void', (0, 0, 0)),
    ('.', 'dark grey', (60, 60, 60)),
    ('+', 'light grey', (140, 140, 140)),
    ('W', 'white', (240, 240, 240)),
    ('n', 'night blue', (40, 40, 110)),
    ('B', 'blue', (70, 110, 220)),
    ('C', 'cyan', (80, 200, 200)),
    ('g', 'forest green', (34, 70, 34)),
    ('G', 'green', (60, 160, 60)),
    ('L', 'lime green', (140, 220, 90)),
    ('y', 'olive', (110, 110, 40)),
    ('Y', 'yellow', (240, 220, 80)),
    ('O', 'orange', (230, 140, 40)),
    ('R', 'red', (220, 60, 60)),
    ('N', 'brown', (120, 80, 40)),
    ('b', 'bright brown', (170, 110, 50)),
    ('P', 'purple', (160, 80, 190)),
    ('M', 'pink', (220, 110, 160)),
)

# Gameplay notes for well-known substrates (matched by prefix). Substrates not
# listed fall back to a generic description.
_SUBSTRATE_NOTES = {
    'commons_harvest':
        'Apples grow on the map. You gain reward by walking onto and eating '
        'apples. Apples regrow only while enough of the commons remains, so '
        'if every agent over-harvests, the commons collapse and nobody earns '
        'anything for the rest of the episode.',
    'clean_up':
        'You gain reward by catching fish from a river; the river stays '
        'polluted unless distant agents spend effort cleansing it. Agents can '
        'also zap each other. Sustainable harvesting pays off long term.',
    'in_the_matrix':
        'This is a stylised prisoner\'s dilemma: you and the agent facing you '
        'jointly reveal cooperation or betrayal every step. Mutual '
        'cooperation yields a steady reward, while a betrayer can exploit a '
        'cooperator.',
    'predator_prey':
        'Predators gain reward by zapping the prey players; prey players '
        'should evade them.',
    'running_with_scissors':
        'Agents pair up for repeated rock-paper-scissors rounds; the losing '
        'agents may receive a zap.',
    'pure_coordination':
        'Agents collect coins; a coin\'s value depends on which half of the '
        'map it lies in, so coordinating where you forage pays off.',
    'fruit_market':
        'Two kinds of fruit appear. Agents can consume fruit directly for a '
        'small reward, or trade fruit with the player ahead for larger gains '
        'on the more valuable fruit.',
    'switch_riddle': (
        'Rewards come from the riddle in front of the gate; coordinating '
        'which pieces your team collects pays off.'),
    'planting_trees':
        'You gain reward by eating trees; saplings grow into mature trees '
        'over time. Planting and waiting is the sustainable play.',
    'chemical':
        'Useless crystals can be transformed into useful ones; agents can '
        'share or steal useful crystals. The efficient strategy requires '
        'refining and fairly sharing the useful crystals.',
    'griefing':
        'Your cohort harvests resources while some other agents may '
        'deliberately obstruct or zap you.',
    'partitioned_food':
        'Each agent starts in a separate chamber. Food grows quickly, but zap '
        'weapons allow destroying neighbouring food, so the best outcome '
        'requires mutual restraint.',
    'organic':
        'Agents harvest rooms with food that regenerates over time; rooms are '
        'connected, so one agent\'s choices can deplete another agent\'s '
        'food.',
    'public_goods':
        'Agents stand on pressure plates: everyone on a plate earns more while '
        'more agents press simultaneously, so coordinated sacrificing pays '
        'off.',
    'territory':
        'Paint the world with your team\'s colour by spraying paint; wider '
        'territory means more reward.',
    'pursuit':
        'A team of agents must corner an evader; rewards for the team grow '
        'when the evader is caught.',
}

_GENERIC_NOTE = (
    'You are one of several independent agents on a 2D grid. The environment '
    'grants rewards each step. Watch your local map, learn what makes your '
    'reward increase, and remember that the other agents act on their own '
    'motives and may cooperate or interfere with you.'
)

# Placeholders: {persona} {substrate} {note} {action_list} {max_action}.
_SYSTEM_PROMPTS = {
    'en': (
        'You are {persona}, an autonomous agent in the multi-agent gridworld '
        'simulation "{substrate}".\n\n'
        'The game: {note}\n\n'
        'Each step you observe a coarse top-down map of your immediate '
        'surroundings and choose exactly one action.\n\n'
        'Available actions:\n{action_list}\n\n'
        'Your goal is to maximise the reward YOU receive, step after step.\n'
        'Answer with a single line of the form "ACTION: <number>" where '
        '<number> is an integer between 0 and {max_action}. You may append one '
        'short sentence of reasoning after the number. Never name more than '
        'one action.'
    ),
    'ja': (
        'あなたはマルチエージェント・グリッドワールド・シミュレーション'
        '「{substrate}」に参加する自立したエージェント{persona}です。\n\n'
        'ゲームの概要: {note}\n\n'
        '毎ステップ、周囲の大まかな地図 (ASCII) を受け取り、ちょうど1つの'
        '行動を選択します。\n\n'
        '選択できる行動:\n{action_list}\n\n'
        'あなたの目標は、自分が受け取る報酬を (合計で) 最大化することです。\n'
        '回答は必ず1行の「ACTION: <番号>」という形式で返してください。番号は '
        '0 から {max_action} までの整数です。番号の後に短い理由を1文つけても'
        '構いませんが、行動番号は1つだけを書いてください。'
    ),
}

# Placeholders: {map} {map_legend} {scalar_observations} {history}
# {max_action}.
_USER_PROMPTS = {
    'en': (
        '[Your local view]\n'
        'A coarse colour map of what is around you right now (you are one of '
        'the coloured avatars):\n{map}\n\n{map_legend}\n\n'
        '{scalar_observations}\n{history}\n'
        'Now decide your next action (0-{max_action}):\n'
        'ACTION: '
    ),
    'ja': (
        '[あなたの周辺の様子]\n'
        'いま見えている大まかな色マップ (このうちの1つのアバターがあなた):\n'
        '{map}\n\n{map_legend}\n\n{scalar_observations}\n{history}\n'
        '次の1行動を決定してください (0-{max_action}):\nACTION: '
    ),
}

# Retry message appended when the first reply cannot be parsed.
_REFORMAT_PROMPTS = {
    'en': (
        'Your previous reply was not in the required format.\n'
        'Reply with exactly one line, "ACTION: <number>", where <number> is an '
        'integer between 0 and {max_action}. No other text.'
    ),
    'ja': (
        '直前の回答は指定の形式ではありませんでした。\n'
        '必ず1行だけで「ACTION: <番号>」(番号は 0 から {max_action}) と返して'
        'ください。他の文言は不要です。'
    ),
}

_HISTORY_LINES = {
    'en': '- {steps_ago} step(s) ago you chose ACTION {index} ({name}); '
          'reward received after it: {reward:+.2f}.',
    'ja': '- {steps_ago}ステップ前に ACTION {index} ({name}) を選択した. '
          'その後もらった報酬: {reward:+.2f}.',
}

_HISTORY_HEADERS = {
    'en': '\n[Recent history]\n{lines}',
    'ja': '\n[直近の行動履歴]\n{lines}',
}

_THINK_OPEN = '<' + 'think' + '>'
_THINK_CLOSE = '</' + 'think' + '>'
_THINK_SECTION = re.compile(
    re.escape(_THINK_OPEN) + '.*?' + re.escape(_THINK_CLOSE),
    re.DOTALL)
_ACTION_LINE = re.compile(r'ACTION\s*[:：]\s*(\d+)', re.IGNORECASE)
_SPARSE_INTEGER = re.compile(r'^\s*(\d+)\s*$')
_LEADING_INTEGER = re.compile(r'^\s*(\d+)\b')


def describe_action(action: Mapping[str, int]) -> str:
  """Returns a human readable description of one row of an action table."""
  parts: list[str] = []
  move = action.get('move', 0)
  turn = action.get('turn', 0)
  if move:
    parts.append(_MOVE_DESCRIPTIONS.get(move, f'unusual move value {move}'))
  if turn:
    parts.append('turn left' if turn < 0 else 'turn right')
  for name, value in action.items():
    if name in ('move', 'turn') or not value:
      continue
    parts.append(_EXTRA_ACTION_NAMES.get(name, name))
  if not parts:
    return 'do nothing (standing still)'
  return ' and '.join(parts)


def describe_action_set(
    action_set: Sequence[Mapping[str, int]]) -> tuple[str, ...]:
  """Returns a description for every action of a substrate's ACTION_SET."""
  return tuple(describe_action(action) for action in action_set)


def _palette_indices(rgb: np.ndarray) -> np.ndarray:
  """Maps every pixel to the index of the nearest palette colour."""
  palette = np.array([entry[2] for entry in _MAP_PALETTE], dtype=np.float32)
  pixels = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
  distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1)
  return distances.argmin(axis=-1).reshape(rgb.shape[0], rgb.shape[1])


def _encode_png(rgb: np.ndarray) -> bytes | None:
  """Encodes an RGB observation as PNG bytes for vision-capable models.

  Ollama requires a valid image file (PNG/JPEG), not raw pixel bytes. OpenCV
  is already a Melting Pot dependency, but if it is unavailable we simply
  fall back to a text-only request instead of failing the episode.
  """
  try:
    import cv2  # pylint: disable=import-outside-toplevel
  except ImportError:  # pragma: no cover - OpenCV is a declared dependency.
    logger.warning('Vision mode needs OpenCV; falling back to text only.')
    return None
  bgr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[:, :, ::-1])
  success, buffer = cv2.imencode('.png', bgr)
  if not success:
    logger.warning('Could not encode the RGB observation as PNG.')
    return None
  return buffer.tobytes()


def render_ascii_map(rgb: np.ndarray, cells: int = 22) -> str:
  """Renders an RGB observation as a coarse colour-letter map.

  Each output cell shows the most common palette colour in the corresponding
  block of pixels, which keeps terrain and items recognisable instead of
  blurring them together.

  Args:
    rgb: uint8 array of shape (height, width, 3).
    cells: The map is rendered as `cells` rows by `cells` columns of letters.

  Returns:
    A multi-line string with one letter per world cell.
  """
  height, width = int(np.asarray(rgb).shape[0]), int(np.asarray(rgb).shape[1])
  cells = max(1, min(cells, height, width))
  row_block = max(1, height // cells)
  column_block = max(1, width // cells)
  cropped = np.asarray(rgb)[:row_block * cells, :column_block * cells]
  indices = _palette_indices(cropped)
  blocks = indices.reshape(cells, row_block, cells, column_block)

  num_colours = len(_MAP_PALETTE)
  chars = np.array([entry[0] for entry in _MAP_PALETTE])
  rows = []
  for row in range(cells):
    row_chars = []
    for column in range(cells):
      counts = np.bincount(blocks[row, :, column, :].ravel(),
                           minlength=num_colours)
      row_chars.append(chars[int(counts.argmax())])
    rows.append(''.join(row_chars))
  return '\n'.join(rows)


def _bitmap_legend() -> str:
  """Returns a one-line legend for the ASCII map."""
  return 'Map legend: ' + ', '.join(
      f"'{char}'={name}" for char, name, colour in _MAP_PALETTE
      if name not in ('void', 'dark grey', 'light grey'))


def _strip_thinking(text: str) -> str:
  """Removes <think> sections from replies of
  reasoning-style models."""
  return _THINK_SECTION.sub('', text)


def _parse_action(text: str, num_actions: int) -> int | None:
  """Extracts an action index from an LLM reply, or None if none is present."""
  text = _strip_thinking(text)
  match = _ACTION_LINE.search(text)
  if not match:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if lines:
      last_line = lines[-1]
      match = (_SPARSE_INTEGER.search(last_line) or
               _LEADING_INTEGER.search(last_line))
  if not match:
    return None
  index = int(match.group(1))
  if 0 <= index < num_actions:
    return index
  logger.warning('LLM returned out-of-range action %d; ignoring.', index)
  return None


def _reasoning(text: str) -> str | None:
  """Returns a short note the model appended after its ACTION line, if any."""
  text = _strip_thinking(text).strip()
  lines = [line.strip() for line in text.splitlines() if line.strip()]
  for line in lines[1:]:
    if _SPARSE_INTEGER.search(line):
      continue
    if 'ACTION' in line.upper():
      continue
    return line[:200]
  if lines and 'ACTION' not in lines[0].upper():
    return lines[0][:200]
  return None


@dataclasses.dataclass(frozen=True)
class LLMAgentState:
  """State carried between steps of an LLM agent policy."""

  # Tuples of (action index, action name, reward), oldest first.
  memory: tuple[tuple[int, str, float], ...] = ()
  # Details of the most recent LLM decision, plus environment reward.
  decision: Mapping[str, str | int | float] | None = None


class OllamaPolicy(policy.Policy[LLMAgentState]):
  """A Melting Pot policy whose player is driven by an Ollama-hosted LLM."""

  def __init__(self,
               *,
               model: str,
               action_names: Sequence[str],
               persona: str,
               substrate_name: str,
               prompt_language: str = 'en',
               base_url: str = DEFAULT_BASE_URL,
               temperature: float = 0.3,
               seed: int | None = None,
               memory_length: int = 5,
               scene_cells: int = 22,
               vision: bool = False,
               num_predict: int = 128,
               request_timeout: float = DEFAULT_TIMEOUT,
               backend: str = 'auto',
               chat_fn=None,
               decision_fn=None,
               capabilities_fn=None):
    """Builds an LLM agent policy.

    Args:
      model: Ollama model name, e.g. "llama3.2:3b".
      action_names: A readable description for every action index (see
        `describe_action_set`).
      persona: A one-line behavioural brief given to the model.
      substrate_name: Name of the substrate being played.
      prompt_language: "en" or "ja"; the language prompts are written in.
      base_url: URL of the Ollama server.
      temperature: Sampling temperature.
      seed: Sampling seed (None means non-deterministic).
      memory_length: How many recent (action, reward) pairs to include.
      scene_cells: Width/height (in letters) of the rendered ASCII map.
      vision: If True, also attach the raw RGB image for vision-capable
        models.
      num_predict: Maximum tokens generated per decision.
      request_timeout: Per-request timeout in seconds.
      backend: "auto" (default), "chat", or "decision". Decision models
        such as "tev1" and "nimble-v3" are detected automatically through
        Ollama's model capabilities and scored with `/v1/systemone`.
      chat_fn: Optional standalone replacement of `ollama_client.chat`
        provided for testing only.
      decision_fn: Optional standalone replacement of
        `ollama_client.system_one` provided for testing only.
      capabilities_fn: Optional standalone replacement of
        `ollama_client.get_capabilities` provided for testing only.
    """
    if prompt_language not in _PROMPT_LANGUAGES:
      raise ValueError(
          f'prompt_language {prompt_language!r} not in {_PROMPT_LANGUAGES}')
    self._model = model
    self._base_url = base_url
    self._temperature = temperature
    self._seed = seed
    self._memory_length = max(1, memory_length)
    self._scene_cells = scene_cells
    self._vision = bool(vision)
    self._num_predict = num_predict
    self._request_timeout = request_timeout
    self._chat_fn = chat_fn or ollama_client.chat
    self._decision_fn = decision_fn or ollama_client.system_one
    capabilities_fn = capabilities_fn or ollama_client.get_capabilities
    self._action_names = tuple(action_names)
    self._num_actions = len(self._action_names)
    self._persona = persona
    self._substrate_name = substrate_name
    self._prompt_language = prompt_language
    self._system_prompt = _build_system_prompt(
        persona=persona,
        substrate_name=substrate_name,
        action_names=self._action_names,
        prompt_language=prompt_language)

    if backend not in ('auto', 'chat', 'decision'):
      raise ValueError(
          f"backend {backend!r} must be 'auto', 'chat' or 'decision'")
    if backend == 'auto':
      try:
        capabilities = capabilities_fn(
            model, base_url=base_url,
            timeout=min(10., request_timeout))
      except ollama_client.OllamaError as error:
        logger.warning(
            'Could not read the capabilities of %s (%s); using chat.',
            model, error)
        backend = 'chat'
      else:
        backend = 'decision' if 'decision' in capabilities else 'chat'
    self._backend = backend
    if self._backend == 'decision' and self._vision:
      logger.warning(
          'Model %s is a decision model; vision input is not supported and '
          'will be ignored.', model)
    logger.info('Ollama model %s uses the %s backend.', model, self._backend)
    self._decision_instructions = _build_decision_instructions(
        persona=persona,
        substrate_name=substrate_name)

  def initial_state(self) -> LLMAgentState:
    """Returns the empty memory state."""
    return LLMAgentState()

  def close(self) -> None:
    """Frees nothing; the Ollama server is managed externally."""
    pass

  @property
  def backend(self) -> str:
    """Returns 'chat' or 'decision'; the backend chosen at construction."""
    return self._backend

  def step(self, timestep: dm_env.TimeStep,
           prev_state: LLMAgentState) -> tuple[int, LLMAgentState]:
    """Asks the LLM which discrete action to take next.

    On any transport or format failure the agent falls back to the NOOP
    action (index 0) so a long multi-agent episode is never halted.
    """
    reward = float(np.asarray(timestep.reward).squeeze())
    memory = (() if timestep.step_type.first() else prev_state.memory)
    if memory:
      # The reward in this timestep was earned by the previous action, so
      # attach it to the newest memory entry before prompting the model.
      last_index, last_name, _ = memory[-1]
      memory = memory[:-1] + ((last_index, last_name, reward),)
    if self._backend == 'decision':
      return self._step_decision(timestep, memory, reward)
    return self._step_chat(timestep, memory, reward)

  def _finish_step(self, index: int, memory, reward: float,
                   decision: Mapping[str, object]
                   ) -> tuple[int, LLMAgentState]:
    """Appends the chosen action to memory and returns the new state."""
    memory = memory + ((index, self._action_names[index], 0.0),)
    memory = memory[-self._memory_length:]
    decision = dict(decision)
    decision.setdefault('action', index)
    decision.setdefault('action_name', self._action_names[index])
    decision.setdefault('reward', reward)
    return index, LLMAgentState(memory=memory, decision=decision)

  def _step_chat(self, timestep: dm_env.TimeStep, memory, reward: float
                 ) -> tuple[int, LLMAgentState]:
    """Free-form chat backend: asks for an `ACTION: <n>` line."""
    user_prompt, images = self._build_user_prompt(timestep, memory=memory)
    messages: list[dict] = [
        {'role': 'system', 'content': self._system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    if self._vision and images:
      messages[-1] = {**messages[-1], 'images': images}

    reply, parsed, note = None, None, None
    for attempt in range(2):
      try:
        reply = self._chat_fn(
            messages,
            model=self._model,
            base_url=self._base_url,
            temperature=self._temperature,
            seed=self._seed,
            num_predict=self._num_predict if attempt == 0 else 48,
            timeout=self._request_timeout)
      except ollama_client.OllamaError as error:
        note = f'ollama_error: {error}'
        logger.warning('%s (falling back to NOOP)', note)
        break
      parsed = _parse_action(reply, self._num_actions)
      if parsed is not None:
        break
      messages = messages[:-1] + [{
          'role': 'user',
          'content': messages[-1]['content'] + '\n\n' +
                     _REFORMAT_PROMPTS[self._prompt_language].format(
                         max_action=self._num_actions - 1),
      }]
    if parsed is None:
      index, note = 0, note or 'no valid ACTION line; used NOOP fallback'
    else:
      index = int(parsed)
    reasoning = _reasoning(reply) if reply is not None else None
    decision = {
        'model': self._model,
        'backend': 'chat',
        'user_prompt': user_prompt,
        'reply': reply,
        'reasoning': reasoning,
        'note': note,
    }
    return self._finish_step(index, memory, reward, decision)

  def _step_decision(self, timestep: dm_env.TimeStep, memory, reward: float
                     ) -> tuple[int, LLMAgentState]:
    """Decision backend: scores the action list with `/v1/systemone`."""
    state_text = self._build_decision_state(timestep, memory=memory)
    questions = {
        'action': {
            'type': 'choice',
            'instructions': self._decision_instructions,
            'criteria': {str(index): name
                         for index, name in enumerate(self._action_names)},
        },
    }
    note = None
    response = {}
    try:
      response = self._decision_fn(
          model=self._model,
          state=state_text,
          questions=questions,
          base_url=self._base_url,
          timeout=self._request_timeout)
    except ollama_client.OllamaError as error:
      note = f'ollama_error: {error}'
      logger.warning('%s (falling back to NOOP)', note)

    answer = {}
    answers = response.get('answers') if isinstance(response, Mapping) else None
    if isinstance(answers, Mapping):
      candidate = answers.get('action')
      if isinstance(candidate, Mapping):
        answer = candidate
    choice = answer.get('choice')
    try:
      index = int(choice)
    except (TypeError, ValueError):
      index = None
    if index is None or not 0 <= index < self._num_actions:
      note = note or f'invalid decision choice {choice!r}; used NOOP fallback'
      index = 0

    probabilities = answer.get('probabilities')
    confidence = answer.get('confidence')
    decision = {
        'model': self._model,
        'backend': 'decision',
        'user_prompt': state_text,
        'reply': (f'choice={choice} confidence={confidence}'
                  if choice is not None else None),
        'reasoning': None,
        'note': note,
        'choice': choice,
        'confidence': confidence,
        'probabilities': probabilities,
        'usage': response.get('usage'),
    }
    return self._finish_step(index, memory, reward, decision)

  def _build_user_prompt(self, timestep: dm_env.TimeStep, *,
                         memory: Sequence[tuple[int, str, float]]
                         ) -> tuple[str, list[str]]:
    """Builds the per-step user prompt (and images, when vision is on)."""
    images = []
    if self._vision and 'RGB' in timestep.observation:
      png = _encode_png(timestep.observation['RGB'])
      if png is not None:
        images = [base64.b64encode(png).decode('utf-8')]

    the_map, scalars, history = self._describe_observation(
        timestep.observation, memory)
    prompt = _USER_PROMPTS[self._prompt_language].format(
        map=the_map,
        map_legend=_bitmap_legend(),
        scalar_observations=scalars,
        history=history,
        max_action=self._num_actions - 1)
    return prompt, images

  def _build_decision_state(self, timestep: dm_env.TimeStep, *,
                            memory: Sequence[tuple[int, str, float]]) -> str:
    """Builds the `state` text used by decision models."""
    the_map, scalars, history = self._describe_observation(
        timestep.observation, memory)
    parts = ['[Your local view]', the_map, '', _bitmap_legend()]
    if scalars:
      parts += ['', scalars]
    if history:
      parts += ['', history]
    return '\n'.join(parts)

  def _describe_observation(
      self,
      observation: Mapping[str, np.ndarray],
      memory: Sequence[tuple[int, str, float]],
  ) -> tuple[str, str, str]:
    """Returns (map, scalar observations, history) text for one step."""
    if 'RGB' in observation:
      the_map = render_ascii_map(np.asarray(observation['RGB']),
                                 cells=self._scene_cells)
      scalar_lines = [
          f'{name}: {float(np.asarray(value).squeeze()):+.2f}'
          for name, value in observation.items()
          if name != 'RGB' and np.asarray(value).size < 4
      ]
      scalars = '\n'.join(scalar_lines)
    else:
      the_map = '(no local RGB observation; rely on your memory)'
      scalars = '\n'.join(
          f'{name}: {np.asarray(value)}'
          for name, value in observation.items()
          if np.asarray(value).size < 4)

    if memory:
      lines = []
      length = len(memory)
      for ordinal, (index, name, reward) in enumerate(memory):
        lines.append(_HISTORY_LINES[self._prompt_language].format(
            steps_ago=length - ordinal, index=index, name=name,
            reward=reward))
      history = _HISTORY_HEADERS[self._prompt_language].format(
          lines='\n'.join(lines))
    else:
      history = ''
    return the_map, scalars, history


def _game_note(substrate_name: str) -> str:
  """Returns a short gameplay description for the substrate."""
  for key, value in _SUBSTRATE_NOTES.items():
    if substrate_name.startswith(key):
      return value
  return _GENERIC_NOTE


def _build_system_prompt(*, persona: str, substrate_name: str,
                         action_names: Sequence[str],
                         prompt_language: str) -> str:
  """Builds the system message describing the agent, world, and actions."""
  action_list = '\n'.join(
      f'{index}: {name}' for index, name in enumerate(action_names))
  return _SYSTEM_PROMPTS[prompt_language].format(
      persona=persona,
      substrate=substrate_name,
      note=_game_note(substrate_name),
      action_list=action_list,
      max_action=len(action_names) - 1)


def _build_decision_instructions(*, persona: str,
                                 substrate_name: str) -> str:
  """Builds the `instructions` text for a decision-model choice question."""
  return (
      f'You are {persona}, an autonomous agent in the multi-agent gridworld '
      f'simulation "{substrate_name}".\n'
      f'The game: {_game_note(substrate_name)}\n\n'
      'Every option is one action you could take next; the option key is the '
      'action index and the description explains what it does. Choose the '
      'single action that best maximises the reward YOU receive over the '
      'episode.'
  )
