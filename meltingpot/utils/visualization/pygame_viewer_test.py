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
"""Tests for the pygame viewer (uses SDL's dummy video driver)."""

import os
import unittest

# Must be set before pygame initialises its video subsystem.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

from absl.testing import absltest
import numpy as np

from meltingpot.utils.visualization import pygame_viewer


def _make_viewer(**kwargs):
  try:
    import pygame  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise unittest.SkipTest('pygame is not installed')
  del pygame
  try:
    return pygame_viewer.PygameViewer(caption='test', fps=1000, **kwargs)
  except RuntimeError as error:
    raise unittest.SkipTest(f'No video device: {error}')


class PygameViewerTest(absltest.TestCase):

  def test_render_rgb_frame(self):
    viewer = _make_viewer(max_width=48, max_height=32)
    self.addCleanup(viewer.close)
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    self.assertTrue(viewer.render(frame, ['step 0', 'rewards: +1.0']))
    self.assertEqual(viewer.window_size, (48, 32))

  def test_render_grayscale_frame(self):
    viewer = _make_viewer(max_width=40, max_height=40)
    self.addCleanup(viewer.close)
    frame = np.zeros((20, 20), dtype=np.uint8)
    self.assertTrue(viewer.render(frame))
    self.assertEqual(viewer.window_size, (40, 40))

  def test_downscales_large_frames(self):
    viewer = _make_viewer(max_width=100, max_height=100)
    self.addCleanup(viewer.close)
    frame = np.zeros((80, 60, 3), dtype=np.uint8)
    self.assertTrue(viewer.render(frame))
    self.assertEqual(viewer.window_size, (60, 80))

  def test_escape_closes_viewer(self):
    viewer = _make_viewer()
    self.addCleanup(viewer.close)
    import pygame  # pylint: disable=g-import-not-at-top
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN,
                                         key=pygame.K_ESCAPE))
    self.assertFalse(viewer.poll())
    self.assertFalse(viewer.render(np.zeros((8, 8, 3), dtype=np.uint8)))


if __name__ == '__main__':
  absltest.main()
