# Using substrates from Python

The public `meltingpot.substrate` module provides the basic functions needed to
inspect and build a substrate. This page shows the usual workflow without
requiring knowledge of the individual substrate configuration files.

## Choose a substrate

Available substrate names are exposed through `SUBSTRATES`:

```python
from meltingpot import substrate

print(sorted(substrate.SUBSTRATES))
```

Once you have chosen a substrate, inspect its configuration before building it:

```python
name = "clean_up"
config = substrate.get_config(name)

print(config.valid_roles)
print(config.default_player_roles)
```

`get_config` returns the locked configuration for that substrate.
`valid_roles` contains the role names accepted by the substrate and
`default_player_roles` provides its standard player assignment. For example,
Clean Up exposes the `default` role and uses seven players by default.

## Build with the default roles

The simplest way to build a substrate is to reuse its default role assignment:

```python
from meltingpot import substrate

name = "clean_up"
config = substrate.get_config(name)

with substrate.build(name, roles=config.default_player_roles) as env:
  timestep = env.reset()
  print(timestep.observation)
```

The environment is a `dm_env`-style environment. Its specifications can be
inspected directly:

```python
with substrate.build(name, roles=config.default_player_roles) as env:
  print(env.observation_spec())
  print(env.action_spec())
  print(env.reward_spec())
```

## Choose player roles

Some substrates support more than one role. Do not infer the allowed strings
from the substrate name. Read them from the configuration:

```python
from meltingpot import substrate

name = "predator_prey__open"
config = substrate.get_config(name)

print("valid roles:", sorted(config.valid_roles))
print("default assignment:", config.default_player_roles)
```

To use another assignment, pass one valid role for each player:

```python
roles = list(config.default_player_roles)
roles[0] = next(role for role in config.valid_roles if role != roles[0])

with substrate.build(name, roles=roles) as env:
  timestep = env.reset()
```

The length of the `roles` sequence determines the number of players passed to
the substrate builder. Every entry should come from `config.valid_roles`.

## Build from a configuration

`build_from_config` is useful when code already has a configuration object. The
roles are still supplied explicitly:

```python
from meltingpot import substrate

config = substrate.get_config("clean_up")

with substrate.build_from_config(
    config, roles=config.default_player_roles
) as env:
  timestep = env.reset()
```

Use `build` when starting from a substrate name. Use `build_from_config` when
starting from a configuration object. For either path, `get_config` is the
place to discover the supported roles and default player assignment.

## Displaying a live GUI

Melting Pot does not open a window by default; `env.step()` only returns numpy
arrays. There are three ways to look at a running simulation.

### Play a substrate yourself

The bundled human-player scripts render the substrate with pygame and read
the keyboard. They require a graphical system (a local desktop, or an X11
forwarded session):

```shell
python meltingpot/human_players/play_clean_up.py
python meltingpot/human_players/play_commons_harvest.py --observation WORLD.RGB
```

Use `--observation RGB` for the first-person view and `--observation
WORLD.RGB` (the default) for the third-person view. `TAB` switches players.

### Live view of an LLM-agent episode

The LLM example ships a `--gui` flag which opens a pygame window and draws
`WORLD.RGB` after every step, with an overlay showing the step number, the
latest actions and the per-player rewards:

```shell
python examples/llm_agents/run_llm_simulation.py \
    --substrate=commons_harvest__open --players=2 --steps=40 \
    --model=llama3.2:3b --gui
```

Close the window or press `ESC` to stop the episode. `--fps` caps the refresh
rate. To watch a single agent's egocentric view instead, use
`--gui_observation=RGB --gui_player=0`.

The same viewer can be used in your own loop
(`meltingpot/utils/visualization/pygame_viewer.py`):

```python
from meltingpot.utils.visualization import pygame_viewer

viewer = pygame_viewer.PygameViewer(caption="my simulation", fps=8)
try:
  timestep = env.reset()
  while viewer.render(
      timestep.observation[0]["WORLD.RGB"],
      [f"reward: {timestep.reward[0]:+.2f}"]):
    timestep = env.step(actions)
finally:
  viewer.close()
```

`render()` returns `False` once the user closes the window. The viewer needs
`pygame-ce` (already a Melting Pot dependency) and a display; on a headless
machine it raises a clear `RuntimeError`, which the example catches by
disabling the GUI with a warning.

### Headless alternatives

- `--save_frames` in the LLM runner writes one PNG per step.
- OpenCV (`cv2.VideoWriter`) can turn `timestep.observation[0]["WORLD.RGB"]`
  frames into a video; see `meltingpot/utils/evaluation/video_subject.py`.
- For remote machines use `ssh -X`/`ssh -Y` (X11 forwarding) so pygame can
  open its window locally.
