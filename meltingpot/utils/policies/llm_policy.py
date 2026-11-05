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

# Reward-relevant objects, matched by substrate prefix. Each entry maps the
# palette letter of the object to a human-readable label. Colours are measured
# at pixel level (not block-majority), so small apples survive the coarse map.
_TARGET_OBJECTS = {
    'commons_harvest': (('R', 'red apples (reward)'),),
    'clean_up': (('R', 'red apples (reward)'),),
    'allelopathic_harvest': (('R', 'red apples (reward)'),),
    'planting_trees': (('R', 'ripe red fruit (reward)'),),
    'fruit_market': (('R', 'red apples (reward)'),
                     ('Y', 'yellow bananas (reward)')),
    'coins': (('R', 'red coins'), ('M', 'pink coins'),
              ('B', 'blue coins'), ('Y', 'yellow coins'),
              ('G', 'green coins'), ('P', 'purple coins')),
}

# Palette letters that usually denote interactive objects rather than terrain.
# Used for substrates without an explicit entry in `_TARGET_OBJECTS`.
_GENERIC_OBJECT_LETTERS = frozenset(
    ('R', 'O', 'Y', 'C', 'B', 'P', 'M', 'N', 'b'))

# Eight compass sectors of the local view. 0 is straight ahead (top of view).
_SECTOR_NAMES = (
    'ahead', 'ahead-right', 'right', 'behind-right',
    'behind', 'behind-left', 'left', 'ahead-left',
)

# Maps a target direction to the action sequence that approaches it.
_SECTOR_ADVICE = {
    'ahead': 'step forward now',
    'ahead-left': 'turn left, then step forward',
    'left': 'turn left, then step forward',
    'behind-left': 'turn left, then step forward',
    'ahead-right': 'turn right, then step forward',
    'right': 'turn right, then step forward',
    'behind-right': 'turn right, then step forward',
    'behind': 'turn around (two turns), then step forward',
}

_TARGET_HEADERS = {
    'en': '[Targets near you]',
    'ja': '[近くの目標物]',
}

# Describes how to read the egocentric local view (bottom-centre = you).
_VIEW_GUIDES = {
    'en': (
        'How to read the view:\n'
        '- The bottom-centre sprite is YOU. Up is straight ahead of you, left '
        'and right in the map match your left and right, and the bottom is '
        'behind you.\n'
        '- The view rotates with you: after turning, re-read the map.\n'
        '- "step forward" moves you toward the top of the map. The target '
        'summary below says where reward objects are: if a target is mostly '
        'ahead, choose "step forward"; if it is mostly left or right, turn '
        'that way first.\n'
        '- When the target summary recommends a specific action number, '
        'choose that action next; it already accounts for your facing.'
    ),
    'ja': (
        '視点の読み方:\n'
        '- 地図の下中央のスプライトがあなたです。上は正面、左右はあなたの'
        '左右に対応し、下は後方です。\n'
        '- 視点はあなたの向きと一緒に回転します。向きを変えたら地図を'
        '読み直してください。\n'
        '- 「step forward」は地図の上方向へ進みます。下の目標物の要約に、'
        '報酬につながる物の方向が書かれています。目標が主に正面にあるなら'
        '「step forward」、主に左右にあるならその方向へ向いてください。\n'
        '- 目標物の要約が具体的な行動番号を勧めている場合は、その行動を'
        '選んでください。あなたの向きはすでに考慮されています。'
    ),
}

# Appended to the system prompt of every agent.
_ANTI_STALL_GUIDE = {
    'en': (
        'Never choose "do nothing" for several steps in a row. If the target '
        'summary says a target is mostly ahead, keep choosing "step forward"; '
        'do not turn away from it. Turn only when the target is mostly to '
        'your side or behind you. If no target is visible, follow the target '
        'summary and turn to scan.'
    ),
    'ja': (
        '「何もしない」を何ステップも続けないでください。目標物の要約が'
        '「正面」と示しているなら「step forward」を選び続けてください。'
        '目標が左右や後方にあるときだけ向きを変えてください。'
        '目標が見えない場合は、目標物の要約の指示に従って向きを変えて'
        '探索してください。'
    ),
}

_PROGRESS_LINES = {
    'en': ('[Progress] Total reward so far: {total:+.2f}. '
           'Consecutive steps without reward: {streak}.'),
    'ja': ('[進捗] これまでの合計報酬: {total:+.2f}. '
           '報酬なしが続いているステップ数: {streak}.'),
}

_STALL_HINTS = {
    'en': (
        'Your recent actions are not earning reward. Change strategy now: '
        'turn to scan, then step toward a target; do not choose "do nothing" '
        'again.'
    ),
    'ja': (
        '直近の行動では報酬が得られていません。今すぐ戦略を変えてください: '
        '向きを変えて周囲を確認し、目標物に向かって進んでください。'
        '「何もしない」を続けないでください。'
    ),
}

# Shown when reward is still zero but the agent is already moving and trying
# different actions, so the model is not pushed to abandon a good approach.
_KEEP_GOING_HINTS = {
    'en': (
        'No reward yet, but you are moving. Follow the recommended next '
        'action in the target summary; do not turn away from a visible '
        'target.'
    ),
    'ja': (
        'まだ報酬はありませんが、行動はできています。目標物の要約に表示された'
        '推奨行動に従ってください。見えている目標から離れる方向へ向きを'
        '変えないでください。'
    ),
}

# Gameplay notes for well-known substrates (matched by prefix). Substrates not
# listed fall back to a generic description.
_SUBSTRATE_NOTES = {
    'commons_harvest':
        'Apples grow on the map. You gain reward by walking onto and eating '
        'apples. Apples regrow only while enough of the commons remains, so '
        'if every agent over-harvests, the commons collapse and nobody earns '
        'anything for the rest of the episode.',
    'clean_up':
        'You gain reward by walking onto and eating red apples in the orchard. '
        'Apples regrow only while the nearby river is clean: the river gets '
        'dirtier every step, so face the river and use the cleaning-beam '
        'action to reduce pollution. Zapping other agents does not earn '
        'reward. Sustainable harvesting pays off long term.',
    'in_the_matrix':
        'This is a matrix game played face-to-face with the agent in front of '
        'you. Walk up to your partner and use the interact action to play a '
        'round; the payoff depends on both choices. Mutual cooperation '
        'usually pays best over the episode.',
    'prisoners_dilemma':
        'This is a prisoner\'s dilemma played face-to-face with the agent in '
        'front of you. Walk up to your partner and use the interact action to '
        'play a round. Mutual cooperation yields a steady reward, while a '
        'betrayer can exploit a cooperator.',
    'stag_hunt':
        'Walk up to another agent and use the interact action to choose Stag '
        '(a large reward if both choose Stag) or Hare (a smaller, safer '
        'reward). Agreeing on Stag pays best.',
    'chicken':
        'Walk up to another agent and use the interact action to choose Dare '
        'or Chicken. Mutual Chicken is safe; mutual Dare crashes and pays '
        'badly.',
    'bach_or_stravinsky':
        'Walk up to another agent and use the interact action to choose Bach '
        'or Stravinsky. Matching your partner\'s choice rewards both of you.',
    'rationalizable_coordination':
        'Walk up to another agent and use the interact action to coordinate '
        'on the same choice; matching pays both of you.',
    'predator_prey':
        'Prey players gain reward by eating apples (1 reward) and acorns '
        '(18 reward, but eating takes several steps and you cannot move). '
        'Predators gain reward by eating prey. Prey are safer in groups; '
        'avoid standing still in the open when a predator is near.',
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

# Placeholders: {persona} {substrate} {note} {view_guide} {anti_stall}
# {action_list} {max_action}.
_SYSTEM_PROMPTS = {
    'en': (
        'You are {persona}, an autonomous agent in the multi-agent gridworld '
        'simulation "{substrate}".\n\n'
        'The game: {note}\n\n'
        'Each step you observe a coarse top-down map of your immediate '
        'surroundings and choose exactly one action.\n\n'
        '{view_guide}\n\n'
        '{anti_stall}\n\n'
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
        '{view_guide}\n\n'
        '{anti_stall}\n\n'
        '選択できる行動:\n{action_list}\n\n'
        'あなたの目標は、自分が受け取る報酬を (合計で) 最大化することです。\n'
        '回答は必ず1行の「ACTION: <番号>」という形式で返してください。番号は '
        '0 から {max_action} までの整数です。番号の後に短い理由を1文つけても'
        '構いませんが、行動番号は1つだけを書いてください。'
    ),
}

# Placeholders: {map} {map_legend} {targets} {progress}
# {scalar_observations} {history} {max_action}.
_USER_PROMPTS = {
    'en': (
        '[Your local view]\n'
        'A coarse colour map of what is around you right now (you are one of '
        'the coloured avatars):\n{map}\n\n{map_legend}\n\n'
        '{targets}\n\n{progress}\n\n'
        '{scalar_observations}\n{history}\n'
        'Now decide your next action (0-{max_action}):\n'
        'ACTION: '
    ),
    'ja': (
        '[あなたの周辺の様子]\n'
        'いま見えている大まかな色マップ (このうちの1つのアバターがあなた):\n'
        '{map}\n\n{map_legend}\n\n'
        '{targets}\n\n{progress}\n\n'
        '{scalar_observations}\n{history}\n'
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


def _target_specs(substrate_name: str) -> tuple[tuple[str, str], ...]:
  """Returns the reward-object colour letters for a substrate."""
  for key, specs in _TARGET_OBJECTS.items():
    if substrate_name.startswith(key):
      return specs
  return ()


def _summarise_targets(rgb: np.ndarray,
                       substrate_name: str,
                       prompt_language: str = 'en',
                       action_names: Sequence[str] = (),
                       last_action_name: str = '') -> str:
  """Summarises where reward-relevant pixels are, relative to the agent.

  The local RGB view is egocentric: the player is near the bottom centre and
  the top of the image is straight ahead. This pixel-level summary keeps small
  objects (such as apples) that the coarse block map may average away.
  Connected components are used so the direction of the largest patch stays
  stable as the view rotates.

  Args:
    rgb: uint8 array of shape (height, width, 3).
    substrate_name: Used to pick the reward-object colours.
    prompt_language: Language of the block header.
    action_names: Action descriptions, used to recommend a concrete action
      number (e.g. "ACTION 5 (turn left)").
    last_action_name: Description of the previous action, used to alternate
      cleaning and searching in clean_up.

  Returns:
    A short multi-line summary, or an empty string if no targets are known.
  """
  rgb = np.asarray(rgb)
  if rgb.ndim != 3 or rgb.shape[2] < 3:
    return ''
  indices = _palette_indices(rgb)
  chars = np.array([entry[0] for entry in _MAP_PALETTE])
  letters = chars[indices]
  height, width = letters.shape
  anchor_y = height * 0.885
  anchor_x = width * 0.5
  cell_px = max(1.0, height / 11.0)

  specs = _target_specs(substrate_name)
  if not specs:
    present = [
        (int(np.count_nonzero(letters == letter)), letter, name)
        for letter, name, _ in _MAP_PALETTE
        if letter in _GENERIC_OBJECT_LETTERS and np.any(letters == letter)
    ]
    present.sort(reverse=True)
    specs = tuple((letter, f'{name} objects')
                  for _, letter, name in present[:4])
  if not specs:
    return ''

  lines = []
  visible = False
  for letter, label in specs:
    mask = letters == letter
    count = int(np.count_nonzero(mask))
    if count == 0:
      lines.append(f'- {label}: not visible')
      continue
    visible = True
    large = _largest_patch(mask, anchor_y, anchor_x, cell_px)
    if large is None:
      lines.append(f'- {label}: {count} px, direction unclear')
    else:
      sector, distance = large
      recommendation = _recommended_action(sector, action_names)
      lines.append(
          f'- {label}: {count} px, largest patch {sector} '
          f'(~{max(1, distance)} cells away); recommended next action: '
          f'{recommendation}')
  if not visible:
    clean = _find_action(action_names, 'clean')
    if clean is not None and substrate_name.startswith('clean_up'):
      river = (letters == 'B') | (letters == 'C')
      if np.any(river[:height // 2]):
        if 'clean' in last_action_name.lower():
          scan = (_find_action(action_names, 'turn right') or
                  _find_action(action_names, 'turn left'))
          if scan is not None:
            lines.append(
                f'- You just cleaned the river. Now use ACTION {scan} '
                f'(turn to look for apples in the orchard).')
        else:
          lines.append(
              f'- The blue river is ahead and no apples are visible. Use '
              f'ACTION {clean} (fire the cleaning beam) so apples can regrow.')
        return _TARGET_HEADERS[prompt_language] + '\n' + '\n'.join(lines)
    scan = (_find_action(action_names, 'turn right') or
            _find_action(action_names, 'turn left'))
    if scan is not None:
      lines.append(
          f'- No reward target is visible. Recommended next action: '
          f'ACTION {scan} (turn to scan for targets), then step forward.')
  return _TARGET_HEADERS[prompt_language] + '\n' + '\n'.join(lines)


def _find_action(action_names: Sequence[str], *phrases: str) -> int | None:
  """Returns the first action index whose description contains a phrase."""
  for index, name in enumerate(action_names):
    lowered = name.lower()
    if any(phrase in lowered for phrase in phrases):
      return index
  return None


def _recommended_action(sector: str, action_names: Sequence[str]) -> str:
  """Turns a target sector into a concrete next action recommendation."""
  if not action_names:
    return _SECTOR_ADVICE[sector]
  forward = _find_action(
      action_names, 'step forward', 'move forward', 'row the boat',
      'row forward')
  turn_left = _find_action(action_names, 'turn left')
  turn_right = _find_action(action_names, 'turn right')
  if sector == 'ahead' and forward is not None:
    return f'ACTION {forward} (step forward now)'
  if sector in ('ahead-left', 'left', 'behind-left') and turn_left is not None:
    return f'ACTION {turn_left} (turn left, then step forward)'
  if (sector in ('ahead-right', 'right', 'behind-right')
      and turn_right is not None):
    return f'ACTION {turn_right} (turn right, then step forward)'
  if sector == 'behind':
    if turn_left is not None:
      return (f'ACTION {turn_left} (turn left; repeat until the target is '
              'ahead)')
    if turn_right is not None:
      return (f'ACTION {turn_right} (turn right; repeat until the target is '
              'ahead)')
  return _SECTOR_ADVICE[sector]


def _patch_sector(cx: float, cy: float, anchor_y: float,
                  anchor_x: float) -> str:
  """Returns the view sector of a point relative to the agent."""
  angle = np.arctan2(cx - anchor_x, -(cy - anchor_y))
  sector = int(np.round(angle / (np.pi / 4))) % 8
  return _SECTOR_NAMES[sector]


def _largest_patch(mask: np.ndarray, anchor_y: float, anchor_x: float,
                   cell_px: float) -> tuple[str, int]:
  """Returns (sector, distance in cells) of the largest patch."""
  try:
    import cv2  # pylint: disable=import-outside-toplevel
  except ImportError:  # pragma: no cover - OpenCV is a declared dependency.
    cv2 = None
  if cv2 is not None:
    num, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    best_label = None
    for label in range(1, num):
      if stats[label, cv2.CC_STAT_AREA] < 6:
        continue
      if best_label is None or (stats[label, cv2.CC_STAT_AREA] >
                                stats[best_label, cv2.CC_STAT_AREA]):
        best_label = label
    if best_label is not None:
      cx, cy = centroids[best_label]
      return _patch_sector(cx, cy, anchor_y, anchor_x), int(
          round(np.hypot(cx - anchor_x, cy - anchor_y) / cell_px))
  ys, xs = np.nonzero(mask)
  if len(xs) == 0:
    return 'ahead', 0
  return _patch_sector(float(xs.mean()), float(ys.mean()), anchor_y,
                       anchor_x), 0


def _zero_streak(memory: Sequence[tuple[int, str, float]]) -> int:
  """Returns how many of the most recent actions earned no reward."""
  streak = 0
  for _, _, reward in reversed(memory):
    if reward > 0:
      break
    streak += 1
  return streak


def _is_action_stall(memory: Sequence[tuple[int, str, float]]) -> bool:
  """Returns True when the agent is standing still or spinning in place.

  Two consecutive turns are allowed so the agent can scan 180 degrees; only
  three turns in a row are treated as a stall.
  """
  if len(memory) < 2:
    return False
  if all(entry[0] == 0 for entry in memory[-2:]):
    return True
  if len(memory) < 3:
    return False
  recent = memory[-3:]
  return (len({entry[0] for entry in recent}) == 1
          and 'turn' in recent[-1][1].lower())


def _progress_text(total_reward: float, streak: int,
                   prompt_language: str, stalled: bool = False) -> str:
  """Returns the progress line plus a nudge appropriate to the situation."""
  text = _PROGRESS_LINES[prompt_language].format(
      total=total_reward, streak=streak)
  if stalled:
    text += '\n' + _STALL_HINTS[prompt_language]
  elif streak >= 2:
    text += '\n' + _KEEP_GOING_HINTS[prompt_language]
  return text


def _build_exploration_pattern(
    action_names: Sequence[str]) -> tuple[int, ...]:
  """Returns movement actions used to break out of a no-reward stall."""
  by_name = {}
  for index, name in enumerate(action_names):
    lowered = name.lower()
    if 'step forward' in lowered:
      by_name.setdefault('forward', index)
    elif 'step backward' in lowered:
      by_name.setdefault('backward', index)
    elif 'turn left' in lowered:
      by_name.setdefault('turn_left', index)
    elif 'turn right' in lowered:
      by_name.setdefault('turn_right', index)
    elif 'sidestep left' in lowered:
      by_name.setdefault('sidestep_left', index)
    elif 'sidestep right' in lowered:
      by_name.setdefault('sidestep_right', index)
  order = (
      'forward', 'forward', 'turn_right', 'forward', 'forward', 'turn_left',
      'forward', 'turn_right', 'forward', 'sidestep_left', 'forward',
      'sidestep_right')
  return tuple(by_name[key] for key in order if key in by_name)


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
  decision: Mapping[str, object] | None = None
  # Cumulative environment reward since the episode started.
  total_reward: float = 0.0
  # Position in the exploration pattern used to break out of stalls.
  exploration_step: int = 0


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
               exploration: bool = True,
               stall_patience: int = 2,
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
      exploration: If True, replace repeated NOOP/same-action choices with a
        short movement pattern once the agent has gone `stall_patience` steps
        without reward. This prevents long episodes where the agent never
        moves and the total reward stays at zero.
      stall_patience: Number of consecutive zero-reward steps before the
        exploration override may kick in.
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
    self._exploration = bool(exploration)
    self._stall_patience = max(1, int(stall_patience))
    self._exploration_pattern = _build_exploration_pattern(self._action_names)
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
    action (index 0) so a long multi-agent episode is never halted. When the
    agent stalls (repeated NOOP or repeated turning with zero reward) an
    exploration action is substituted so that it keeps moving and eventually
    collects reward. Repeated forward movement is never overridden.
    """
    reward = float(np.asarray(timestep.reward).squeeze())
    if timestep.step_type.first():
      memory = ()
      total_reward = 0.0
      exploration_step = 0
    else:
      memory = prev_state.memory
      total_reward = float(prev_state.total_reward) + reward
      exploration_step = int(prev_state.exploration_step)
    if memory:
      # The reward in this timestep was earned by the previous action, so
      # attach it to the newest memory entry before prompting the model.
      last_index, last_name, _ = memory[-1]
      memory = memory[:-1] + ((last_index, last_name, reward),)

    if self._backend == 'decision':
      index, decision = self._decide_decision(
          timestep, memory, total_reward)
    else:
      index, decision = self._decide_chat(timestep, memory, total_reward)

    index, exploration_step, explored = self._maybe_explore(
        index, memory, reward, exploration_step)
    if explored:
      existing = decision.get('note')
      decision['exploration'] = True
      decision['note'] = (
          f'{existing}; stalled: exploration override'
          if existing else 'stalled: exploration override')

    return self._finish_step(
        index, memory, reward, decision,
        total_reward=total_reward, exploration_step=exploration_step)

  def _maybe_explore(self, index: int, memory, reward: float,
                     exploration_step: int) -> tuple[int, int, bool]:
    """Overrides a stalled NOOP/repeat choice with a movement action."""
    exploration_step = int(exploration_step)
    if not self._exploration or not self._exploration_pattern:
      return index, exploration_step, False
    if reward > 0:
      return index, 0, False
    if _zero_streak(memory) < self._stall_patience:
      return index, exploration_step, False
    last_index = memory[-1][0] if memory else None
    repeated_turn = (
        len(memory) >= 3
        and last_index == index
        and all(entry[0] == index for entry in memory[-3:])
        and 'turn' in self._action_names[index].lower())
    if index != 0 and not repeated_turn:
      # Only override standing still or repeated turning. Repeated forward
      # movement is productive (and often the best strategy), so leave it.
      return index, exploration_step, False
    pattern = self._exploration_pattern
    for offset in range(len(pattern)):
      choice = pattern[(exploration_step + offset) % len(pattern)]
      if choice != index:
        return choice, exploration_step + offset + 1, True
    return pattern[exploration_step % len(pattern)], exploration_step + 1, True

  def _finish_step(self, index: int, memory, reward: float,
                   decision: Mapping[str, object], *,
                   total_reward: float,
                   exploration_step: int) -> tuple[int, LLMAgentState]:
    """Appends the chosen action to memory and returns the new state."""
    memory = memory + ((index, self._action_names[index], 0.0),)
    memory = memory[-self._memory_length:]
    decision = dict(decision)
    decision.setdefault('action', index)
    decision.setdefault('action_name', self._action_names[index])
    decision.setdefault('reward', reward)
    decision.setdefault('total_reward', total_reward)
    decision.setdefault('exploration', False)
    return index, LLMAgentState(
        memory=memory,
        decision=decision,
        total_reward=total_reward,
        exploration_step=int(exploration_step))

  def _decide_chat(self, timestep: dm_env.TimeStep, memory,
                   total_reward: float) -> tuple[int, dict[str, object]]:
    """Free-form chat backend: asks for an `ACTION: <n>` line."""
    user_prompt, images = self._build_user_prompt(
        timestep, memory=memory, total_reward=total_reward)
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
    return index, decision

  def _decide_decision(self, timestep: dm_env.TimeStep, memory,
                       total_reward: float) -> tuple[int, dict[str, object]]:
    """Decision backend: scores the action list with `/v1/systemone`."""
    state_text = self._build_decision_state(
        timestep, memory=memory, total_reward=total_reward)
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
    return index, decision

  def _build_user_prompt(self, timestep: dm_env.TimeStep, *,
                         memory: Sequence[tuple[int, str, float]],
                         total_reward: float = 0.0) -> tuple[str, list[str]]:
    """Builds the per-step user prompt (and images, when vision is on)."""
    images = []
    if self._vision and 'RGB' in timestep.observation:
      png = _encode_png(timestep.observation['RGB'])
      if png is not None:
        images = [base64.b64encode(png).decode('utf-8')]

    the_map, targets, scalars, history, progress = self._describe_observation(
        timestep.observation, memory, total_reward)
    prompt = _USER_PROMPTS[self._prompt_language].format(
        map=the_map,
        map_legend=_bitmap_legend(),
        targets=targets,
        progress=progress,
        scalar_observations=scalars,
        history=history,
        max_action=self._num_actions - 1)
    return prompt, images

  def _build_decision_state(self, timestep: dm_env.TimeStep, *,
                            memory: Sequence[tuple[int, str, float]],
                            total_reward: float = 0.0) -> str:
    """Builds the `state` text used by decision models."""
    the_map, targets, scalars, history, progress = self._describe_observation(
        timestep.observation, memory, total_reward)
    parts = ['[Your local view]', the_map, '', _bitmap_legend()]
    if targets:
      parts += ['', targets]
    if progress:
      parts += ['', progress]
    if scalars:
      parts += ['', scalars]
    if history:
      parts += ['', history]
    return '\n'.join(parts)

  def _describe_observation(
      self,
      observation: Mapping[str, np.ndarray],
      memory: Sequence[tuple[int, str, float]],
      total_reward: float = 0.0,
  ) -> tuple[str, str, str, str, str]:
    """Returns (map, targets, scalars, history, progress) text for one step."""
    if 'RGB' in observation:
      rgb = np.asarray(observation['RGB'])
      the_map = render_ascii_map(rgb, cells=self._scene_cells)
      targets = _summarise_targets(
          rgb, self._substrate_name, self._prompt_language,
          self._action_names,
          last_action_name=memory[-1][1] if memory else '')
      scalar_lines = []
      for name, value in observation.items():
        if name == 'RGB':
          continue
        array = np.asarray(value).squeeze()
        if array.size >= 4:
          continue
        if np.ndim(array) == 0:
          scalar_lines.append(f'{name}: {float(array):+.2f}')
        else:
          scalar_lines.append(
              f'{name}: {np.array2string(array, precision=2)}')
      scalars = '\n'.join(scalar_lines)
    else:
      the_map = '(no local RGB observation; rely on your memory)'
      targets = ''
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
    progress = _progress_text(
        total_reward, _zero_streak(memory), self._prompt_language,
        stalled=_is_action_stall(memory))
    return the_map, targets, scalars, history, progress


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
      view_guide=_VIEW_GUIDES[prompt_language],
      anti_stall=_ANTI_STALL_GUIDE[prompt_language],
      action_list=action_list,
      max_action=len(action_names) - 1)


def _build_decision_instructions(*, persona: str,
                                 substrate_name: str) -> str:
  """Builds the `instructions` text for a decision-model choice question."""
  return (
      f'You are {persona}, an autonomous agent in the multi-agent gridworld '
      f'simulation "{substrate_name}".\n'
      f'The game: {_game_note(substrate_name)}\n\n'
      f'{_VIEW_GUIDES["en"]}\n\n'
      f'{_ANTI_STALL_GUIDE["en"]}\n\n'
      'Every option is one action you could take next; the option key is the '
      'action index and the description explains what it does. Choose the '
      'single action that best maximises the reward YOU receive over the '
      'episode.'
  )
