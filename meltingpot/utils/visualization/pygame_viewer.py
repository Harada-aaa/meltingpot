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
"""A small pygame window for viewing Melting Pot frames as they are played.

This is a reusable, simulation-loop-friendly alternative to
`meltingpot.human_players.level_playing_utils.run_episode`: it does not own
the environment or the action mapping, it just draws `WORLD.RGB` (or an
individual player's `RGB` view) and reports whether the user closed the
window.

Pygame is imported lazily so that headless simulations do not need it.
"""

import logging
import os
import sys
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class PygameViewer:
  """Draws RGB frames in a resizable pygame window with a text overlay."""

  def __init__(self,
               caption: str = 'Melting Pot',
               fps: int = 8,
               max_width: int = 960,
               max_height: int = 720,
               font_size: int = 20):
    """Opens a pygame window.

    Args:
      caption: Window title.
      fps: Maximum refresh rate in frames per second.
      max_width: Maximum window width; larger frames are downscaled.
      max_height: Maximum window height; larger frames are downscaled.
      font_size: Size of the overlay font.

    Raises:
      RuntimeError: if pygame is not installed or no display is available.
    """
    # pygame/SDL can crash (segfault) instead of raising when there is no
    # video device at all, so refuse to initialise in that case on Linux.
    if (sys.platform.startswith('linux') and
        not os.environ.get('DISPLAY') and
        not os.environ.get('WAYLAND_DISPLAY') and
        not os.environ.get('SDL_VIDEODRIVER')):
      raise RuntimeError(
          'No graphical display found. Run on a desktop or forward X11 '
          'with "ssh -X", or omit --gui to run headless.')
    try:
      import pygame  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - pygame is a dependency.
      raise RuntimeError(
          'The GUI requires pygame; install it with "pip install pygame-ce".'
      ) from error
    try:
      pygame.init()
      pygame.display.set_caption(caption)
      self._font = pygame.font.SysFont(None, font_size)
      self._clock = pygame.time.Clock()
    except pygame.error as error:
      raise RuntimeError(
          'Could not open a pygame window (is a display available?): '
          f'{error}') from error
    self._pygame = pygame
    self._fps = fps
    self._max_size = (max_width, max_height)
    self._screen = None
    self._scale = 1
    self._source_size = None
    self._closed = False
    self._quit_requested = False

  @property
  def window_size(self) -> tuple[int, int] | None:
    """Returns the current (width, height) of the window, if opened."""
    return None if self._screen is None else self._screen.get_size()

  def poll(self) -> bool:
    """Processes window events; returns False if the user asked to quit."""
    if self._closed or self._quit_requested:
      return False
    for event in self._pygame.event.get():
      if event.type == self._pygame.QUIT:
        self._quit_requested = True
        return False
      if (event.type == self._pygame.KEYDOWN and
          event.key == self._pygame.K_ESCAPE):
        self._quit_requested = True
        return False
    return True

  def render(self,
             frame: np.ndarray,
             text_lines: Sequence[str] = ()) -> bool:
    """Draws one RGB frame and returns False when the window was closed.

    Args:
      frame: uint8 array of shape (height, width, 3) in RGB order. Grayscale
        (height, width) frames are expanded to RGB.
      text_lines: Optional lines drawn in a translucent panel at the bottom.
    """
    if not self.poll():
      return False
    try:
      return self._render_impl(frame, text_lines)
    except self._pygame.error as error:
      raise RuntimeError(f'Could not draw the GUI frame: {error}') from error

  def _render_impl(self,
                   frame: np.ndarray,
                   text_lines: Sequence[str]) -> bool:
    """Draws one frame; pygame errors are re-raised by `render`."""
    pygame = self._pygame
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim == 2:
      frame = np.repeat(frame[:, :, None], 3, axis=2)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    scale = max(1, min(self._max_size[1] // height,
                       self._max_size[0] // width))

    if self._screen is None or self._source_size != (height, width):
      self._screen = pygame.display.set_mode((width * scale, height * scale))
      self._source_size = (height, width)
      self._scale = scale

    # Pygame uses column-major surfaces: (H, W, 3) -> (W, H, 3).
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if scale != 1:
      surface = pygame.transform.scale(
          surface, (width * self._scale, height * self._scale))
    self._screen.blit(surface, (0, 0))
    if text_lines:
      self._draw_overlay(text_lines)
    pygame.display.flip()
    self._clock.tick(self._fps)
    return True

  def _draw_overlay(self, text_lines: Sequence[str]) -> None:
    """Draws a translucent, text-only panel along the bottom of the window."""
    pygame = self._pygame
    rendered = [self._font.render(line, True, (255, 255, 255))
                for line in text_lines]
    if not rendered:
      return
    line_height = max(image.get_height() for image in rendered) + 2
    panel_height = line_height * len(rendered) + 8
    panel = pygame.Surface((self._screen.get_width(), panel_height),
                           pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    y = 4
    for image in rendered:
      panel.blit(image, (8, y))
      y += line_height
    self._screen.blit(panel, (0, self._screen.get_height() - panel_height))

  def close(self) -> None:
    """Closes the window and shuts pygame down."""
    if not self._closed:
      self._closed = True
      self._pygame.quit()
