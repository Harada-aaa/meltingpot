# LLM agents for Melting Pot (Ollama backend)

This example turns every player slot of a Melting Pot substrate into an
autonomous **LLM agent** served by a local [Ollama](https://ollama.com)
instance. Agents perceive their local RGB view as a coarse ASCII colour map,
remember their recent actions and rewards, and reply with a discrete action
index that is sent straight to the substrate.

```
 RGB observation ──► ASCII colour map ─┐
 recent actions + rewards ─────────────┤──► prompt ──► Ollama ──► "ACTION: n"
                                       │                          │
                              OllamaPolicy (meltingpot/utils/       │
                              policies/llm_policy.py)              │
                                                                   ▼
                    Melting Pot substrate  ◄── env.step([a0, a1, ...])
```

## Requirements

- A working Melting Pot installation (see the repository README).
- Ollama running locally, e.g. `ollama serve`.
- At least one chat model pulled:

  ```shell
  ollama pull llama3.2:3b        # fast text-only default
  ollama pull gemma3:4b          # supports vision (--vision)
  ```

Check what is available:

```shell
ollama list
```

## Quick start

From the repository root, using the project's virtualenv:

```shell
# 2 LLM agents playing commons_harvest for 40 steps
python examples/llm_agents/run_llm_simulation.py \
    --substrate=commons_harvest__open \
    --players=2 --steps=40 \
    --model=llama3.2:3b \
    --output_dir=/tmp/llm_episode
```

Useful variants:

```shell
# Japanese prompts
... --prompt_language=ja

# Vision-capable model: also send the raw 88x88 RGB image
... --model=gemma3:4b --vision

# Different substrate and population size
python examples/llm_agents/run_llm_simulation.py \
    --substrate=clean_up --players=3 --steps=30 --model=llama3.2:3b

# Save the global view as PNG frames (requires OpenCV)
... --save_frames --output_dir=/tmp/llm_episode
```

## What the agent sees

Each step the policy builds a user message containing:

- a `cells` x `cells` letter map (default 20x20) rendered from the local
  `RGB` observation. Every map cell shows the **dominant** palette colour of
  the corresponding pixel block (not an average), so terrain and items stay
  recognisable;
- a legend for the map characters;
- small scalar observations (e.g. `READY_TO_SHOOT`, `COLLECTIVE_REWARD`);
- the last `--memory` actions with the reward each one earned.

The system message describes the substrate (a curated note for common
substrates, a generic note otherwise), the full action list derived from the
substrate's `ACTION_SET`, and the required output format:

```
ACTION: <number>
```

Replies are parsed defensively: reasoning-model `think` blocks are stripped,
up to two attempts are made, and any transport/format failure falls back to
the NOOP action (index 0) so an episode never halts. Every decision also
records the raw reply and reasoning for the log.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--substrate` | `commons_harvest__open` | Any substrate name. |
| `--players` | substrate default | Number of players (1..len(default roles)). |
| `--model` | `llama3.2:3b` | Ollama model name. |
| `--base_url` | `http://localhost:11434` | Ollama server URL. |
| `--temperature` | `0.3` | Sampling temperature. |
| `--seed` | unset | Sampling seed (unset = varying runs). |
| `--steps` | `40` | Episode length. |
| `--memory` | `5` | Recent (action, reward) pairs in the prompt. |
| `--scene_cells` | `20` | ASCII map size. |
| `--vision` | off | Attach the RGB image (needs a vision model). |
| `--num_predict` | `128` | Max generated tokens per decision. |
| `--prompt_language` | `en` | `en` or `ja`. |
| `--output_dir` | unset | Write `episode.jsonl` (and frames) per run. |
| `--save_frames` | off | Save global view PNGs. |

## Logs

With `--output_dir` set, one JSONL record per step is written:

```json
{
  "substrate": "commons_harvest__open",
  "step": 3,
  "agents": [
    {
      "player": 0,
      "action": 1,
      "action_name": "step forward",
      "latency_s": 0.42,
      "reward": 0.0,
      "note": null,
      "reasoning": "...",
      "reply": "ACTION: 1 ...",
      "user_prompt": "..."
    }
  ],
  "rewards_per_player": [0.0, 0.0]
}
```

## Using the policy in your own loop

`OllamaPolicy` implements the standard `meltingpot.utils.policies.policy.Policy`
interface, so it can be mixed with scripted or pretrained bots:

```python
import meltingpot.bot
from meltingpot import substrate
from meltingpot.utils.policies import llm_policy

config = substrate.get_config("commons_harvest__open")
env = substrate.build("commons_harvest__open", roles=config.default_player_roles[:2])

agent = llm_policy.OllamaPolicy(
    model="llama3.2:3b",
    action_names=llm_policy.describe_action_set(config.action_set),
    persona="a cooperative harvester",
    substrate_name="commons_harvest__open",
)
noop = meltingpot.bot.build("noop_bot")

state = agent.initial_state()
noop_state = noop.initial_state()
timestep = env.reset()

for _ in range(40):
    action, state = agent.step(
        timestep._replace(observation=timestep.observation[0],
                          reward=timestep.reward[0]), state)
    noop_action, noop_state = noop.step(
        timestep._replace(observation=timestep.observation[1],
                          reward=timestep.reward[1]), noop_state)
    timestep = env.step([action, noop_action])
```

The same object can be passed into `meltingpot.scenario` through its `bots`
mapping when running scripted scenarios.

## Notes and limitations

- Melting Pot substrates have **no chat action**; agents cannot exchange free
  text inside the environment. Communication is limited to what the
  environment exposes (movement, zaps, collective reward, ...). Use the
  per-agent `persona` to study heterogeneous behaviour instead.
- Small local models (1-4B) work but often repeat one action. Increase
  `--memory` or use a larger model for richer behaviour. Vision models can be
  slow per step; start with 2 players.
- The ASCII map is egocentric and does not mark which avatar is "you".
  Substrates that expose symbolic observations (`POSITION`, `ORIENTATION`)
  can be added to `_build_user_prompt` for stronger spatial reasoning.
- Latency is dominated by Ollama; each step performs one call per player.

## Tests

The policy has network-free unit tests (mocked Ollama client):

```shell
python -m pytest meltingpot/utils/policies/llm_policy_test.py
```

## 日本語クイックスタート

```shell
# Ollama を起動し、モデルを取得 (別ターミナル)
ollama serve
ollama pull llama3.2:3b

# 2体のLLMエージェントで commons_harvest を実行
python examples/llm_agents/run_llm_simulation.py \
    --substrate=commons_harvest__open --players=2 --steps=40 \
    --model=llama3.2:3b --output_dir=/tmp/llm_episode

# 日本語プロンプト + 視覚モデル
python examples/llm_agents/run_llm_simulation.py \
    --substrate=clean_up --players=3 --steps=20 \
    --model=gemma3:4b --vision --prompt_language=ja
```

- 各エージェントは毎ステップ、周囲のASCII色マップ・直近の行動と報酬・
  行動一覧を受け取り、`ACTION: <番号>` の形式で行動を返します。
- Ollamaが落ちている/応答が壊れている場合でも、NOOP(何もしない)に
  フォールバックしてシミュレーションを継続します。
- ログは `--output_dir` 以下に `episode.jsonl` として保存され、
  各エージェントのプロンプト・生応答・理由・報酬が記録されます。
