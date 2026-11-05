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
"""Multi-agent simulation runner where every player is an LLM agent.

Every player slot of the chosen substrate is driven by an `OllamaPolicy`
(see `meltingpot/utils/policies/llm_policy.py`). Each agent:

1. receives a coarse ASCII rendering of its local RGB view plus a fragment
   of memory (recent actions and rewards),
2. asks the locally running Ollama server what to do next,
3. answers with a discrete action index that is sent to the substrate.

The episode is logged to console and to a per-step JSONL file, and the
global view can optionally be saved as PNG frames.

Usage (from the repository root, with Ollama serving e.g. llama3.2:3b):
  python examples/llm_agents/run_llm_simulation.py \
      --substrate=commons_harvest__open --players=3 --steps=40 \
      --model=llama3.2:3b
"""

import json
import os
import random
import time

# This suppresses most TensorFlow logging; some warnings stay unavoidable.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from absl import logging  # noqa: E402
import numpy as np  # noqa: E402

from meltingpot import substrate  # noqa: E402
from meltingpot.utils.policies import llm_policy  # noqa: E402
from meltingpot.utils.policies.ollama_client import DEFAULT_BASE_URL  # noqa: E402
from meltingpot.utils.visualization import pygame_viewer  # noqa: E402

DEFAULT_SUBSTRATE = "commons_harvest__open"

_PERSONAS = (
    "Forester, a cooperative agent who prefers sustainable harvesting and "
    "avoids needless conflict",
    "Blaze, a bold risk-taker who is happy to act alone and fight when "
    "resources grow scarce",
    "Ledger, a cautious accountant-like agent that only acts when it sees a "
    "clear benefit",
    "Echo, a sociable agent that tries to match the behaviour of whoever is "
    "nearby",
    "Void, a patient observer that watches others and exploits opportunities "
    "late in the episode",
    "Anchor, a steadfast agent that follows a strict personal code whatever "
    "the others do",
    "Sparrow, a fast-moving opportunist that reacts instantly to any change "
    "in the map",
)

FLAGS = flags.FLAGS

flags.DEFINE_string("substrate", DEFAULT_SUBSTRATE, "Substrate to play.")
flags.DEFINE_integer("players", None,
                     "Number of players; defaults to the substrate standard.")
flags.DEFINE_string("model", "llama3.2:3b", "Ollama model for the agents.")
flags.DEFINE_enum("backend", "auto", ["auto", "chat", "decision"],
                  "Ollama API to use. 'auto' detects decision models such as "
                  "tev1 and nimble-v3 from their capabilities.")
flags.DEFINE_string("base_url", DEFAULT_BASE_URL, "Ollama server base URL.")
flags.DEFINE_float("temperature", 0.3, "LLM sampling temperature.")
flags.DEFINE_integer("seed", None,
                     "Sampling seed; omit for varying runs.")
flags.DEFINE_integer("steps", 40, "Number of steps per episode.")
flags.DEFINE_integer("memory", 5,
                     "How many past actions each agent remembers.")
flags.DEFINE_integer("scene_cells", 20, "Width/height of the ASCII map.")
flags.DEFINE_bool("vision", False,
                  "Also attach the RGB image (requires a vision model).")
flags.DEFINE_integer("num_predict", 128,
                     "Maximum tokens generated per agent decision.")
flags.DEFINE_enum("prompt_language", "en", ["en", "ja"],
                  "Language the prompts are written in.")
flags.DEFINE_string("output_dir", "",
                    "Directory for JSONL logs; empty disables files.")
flags.DEFINE_bool("save_frames", False,
                  "Save the global view as PNG images (requires OpenCV).")
flags.DEFINE_bool("gui", False,
                  "Show a live pygame window while the agents play.")
flags.DEFINE_integer("fps", 8, "Maximum GUI refresh rate in frames/second.")
flags.DEFINE_enum("gui_observation", "WORLD.RGB", ["WORLD.RGB", "RGB"],
                  "Show the global world view, or the egocentric view of "
                  "--gui_player.")
flags.DEFINE_integer("gui_player", 0,
                     "Player whose RGB view is displayed when "
                     "--gui_observation=RGB.")


def _overlay_lines(step_number, actions, rewards, totals, action_names):
  """Builds the text overlay shown at the bottom of the GUI window."""
  if actions:
    entries = [f"P{player}:{action_names[action][:18]}"
               for player, action in enumerate(actions)]
    shown = " ".join(entries[:4])
    if len(entries) > 4:
      shown += f" +{len(entries) - 4} more"
  else:
    shown = "(waiting for the first decisions)"
  if rewards is None:
    reward_line = "step rewards: -"
  else:
    reward_line = "step rewards: " + " ".join(f"{reward:+.1f}"
                                              for reward in rewards)
  total_line = "totals: " + " ".join(f"{total:+.1f}" for total in totals)
  return [
      f"step {step_number} | {shown}",
      f"{reward_line} | {total_line}",
      "close the window or press ESC to stop",
  ]


def _ollama_models() -> tuple[str, ...]:
  """Tries to list model names on the configured Ollama server."""
  from meltingpot.utils.policies import ollama_client  # noqa: E402,F401
  try:
    return ollama_client.list_models(FLAGS.base_url)
  except ollama_client.OllamaError as error:
    logging.warning("Could not query Ollama: %s", error)
    return ()


def _save_frame(frame: np.ndarray, directory: str, step: int):
  """Saves the global RGB view as a PNG, if OpenCV is installed."""
  try:
    import cv2  # pylint: disable=import-outside-toplevel
  except ImportError:
    logging.warning("--save_frames ignored: OpenCV (opencv-python) missing.")
    return
  filename = os.path.join(directory, f"frame_{step:05d}.png")
  cv2.imwrite(filename, frame[:, :, ::-1])

def main(_):
  substrate_name = FLAGS.substrate
  config = substrate.get_config(substrate_name)
  max_players = len(config.default_player_roles)
  num_players = FLAGS.players or max_players
  if not 1 <= num_players <= max_players:
    raise ValueError(
        f"{substrate_name} supports between 1 and {max_players} players, "
        f"but --players={num_players} was requested.")
  roles = list(config.default_player_roles)[:num_players]

  logging.set_verbosity(logging.INFO)
  logging.info("Ollama models available: %s",
               ", ".join(_ollama_models()) or "<none>")
  logging.info("Substrate %s with %d LLM agents", substrate_name,
               num_players)

  episode_started = time.time()
  env = substrate.build(substrate_name, roles=roles)
  num_actions = env.action_spec()[0].num_values
  action_names = llm_policy.describe_action_set(config.action_set)
  logging.info("Action set (%d actions): %s", num_actions,
               "; ".join(f"{i}:{n}" for i, n in enumerate(action_names)))

  agents = []
  personas = list(_PERSONAS)
  random.shuffle(personas)
  personas = personas[:num_players]
  while len(personas) < num_players:
    personas.append(f"Agent {len(personas)}, a pragmatic generalist agent")
  for player, persona in enumerate(personas):
    logging.info("player %d persona: %s", player, persona)
    agents.append(llm_policy.OllamaPolicy(
        model=FLAGS.model,
        action_names=action_names,
        persona=persona,
        substrate_name=substrate_name,
        prompt_language=FLAGS.prompt_language,
        base_url=FLAGS.base_url,
        temperature=FLAGS.temperature,
        seed=FLAGS.seed,
        memory_length=FLAGS.memory,
        scene_cells=FLAGS.scene_cells,
        vision=FLAGS.vision,
        num_predict=FLAGS.num_predict,
        backend=FLAGS.backend))
  states = [agent.initial_state() for agent in agents]
  logging.info("LLM backends: %s",
               ", ".join(f"P{player}={agent.backend}"
                         for player, agent in enumerate(agents)))
  if FLAGS.vision and any(agent.backend == "decision" for agent in agents):
    logging.warning("Decision models ignore --vision; no images are sent.")

  log_file = None
  frames_dir = None
  if FLAGS.output_dir:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(FLAGS.output_dir,
                           f"{substrate_name}__{FLAGS.model.replace(':', '_')}"
                           f"__{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "episode.jsonl"), "w",
                    encoding="utf-8")
    if FLAGS.save_frames:
      frames_dir = os.path.join(log_dir, "frames")
      os.makedirs(frames_dir, exist_ok=True)
    logging.info("Writing logs to %s", log_dir)

  total_rewards = np.zeros(num_players)
  timestep = env.reset()

  viewer = None
  if FLAGS.gui:
    try:
      viewer = pygame_viewer.PygameViewer(
          caption=f"Melting Pot: {substrate_name}", fps=FLAGS.fps)
      logging.info("GUI window opened (close it or press ESC to stop).")
    except RuntimeError as error:
      logging.warning("--gui disabled: %s", error)

  previous_actions = None
  previous_rewards = None
  episode_log = []
  try:
    for step_number in range(FLAGS.steps):
      if viewer is not None:
        frame = (timestep.observation[FLAGS.gui_player]["RGB"]
                 if FLAGS.gui_observation == "RGB"
                 else timestep.observation[0]["WORLD.RGB"])
        try:
          keep_running = viewer.render(
              frame,
              _overlay_lines(step_number, previous_actions, previous_rewards,
                             total_rewards, action_names))
        except RuntimeError as error:
          logging.warning("GUI failed (%s); continuing headless.", error)
          viewer.close()
          viewer = None
        else:
          if not keep_running:
            logging.info("GUI window closed; stopping the episode.")
            break
      if frames_dir is not None:
        _save_frame(timestep.observation[0]["WORLD.RGB"], frames_dir,
                    step_number)
      actions = []
      step_log = {
          "substrate": substrate_name,
          "step": step_number,
          "agents": [],
      }
      for player, (agent, state) in enumerate(zip(agents, states)):
        player_timestep = timestep._replace(
            observation=timestep.observation[player],
            reward=timestep.reward[player])
        started = time.time()
        action, state = agent.step(player_timestep, state)
        latency = time.time() - started
        states[player] = state
        actions.append(action)
        decision = dict(state.decision or {})
        step_log["agents"].append({
            "player": player,
            "action": action,
            "action_name": action_names[action],
            "backend": decision.get("backend"),
            "latency_s": round(latency, 2),
            "reward": decision.get("reward"),
            "note": decision.get("note"),
            "reasoning": decision.get("reasoning"),
            "reply": decision.get("reply"),
            "choice": decision.get("choice"),
            "confidence": decision.get("confidence"),
            "probabilities": decision.get("probabilities"),
            "user_prompt": decision.get("user_prompt"),
        })
        logging.info("step %3d | player %d -> %-3d %s%s", step_number, player,
                     action, action_names[action],
                     f"  [{decision.get('note')}]" if decision.get("note")
                     else "")
      timestep = env.step(actions)
      rewards = [float(reward) for reward in timestep.reward]
      total_rewards += np.asarray(rewards)
      previous_actions = actions
      previous_rewards = rewards
      step_log["rewards_per_player"] = [round(reward, 4) for reward in rewards]
      episode_log.append(step_log)
      if log_file is not None:
        log_file.write(json.dumps(step_log, ensure_ascii=False) + "\n")
        log_file.flush()
      if timestep.step_type.last():
        break
  finally:
    if viewer is not None:
      viewer.close()
    if log_file is not None:
      log_file.close()
    for agent in agents:
      agent.close()

  logging.info("Episode finished in %.1fs. Total reward per player: %s",
               time.time() - episode_started, total_rewards.tolist())


if __name__ == "__main__":
  app.run(main)
