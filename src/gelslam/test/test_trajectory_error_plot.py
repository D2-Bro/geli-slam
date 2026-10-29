"""Plot regressions for connecting all stored poses and matched GT samples."""

import numpy as np
import pytest

from gelslam.core.trajectory_error_plot import (
    write_error_colormap,
    write_velocity_component_colormaps,
)


@pytest.mark.parametrize('components', (False, True))
def test_plots_connect_all_samples_across_time_gaps_and_trial_changes(
        tmp_path, monkeypatch, components):
    from matplotlib.figure import Figure

    figures = []
    monkeypatch.setattr(Figure, 'savefig', lambda figure, *a, **kw: figures.append(figure))
    positions = np.array(((0., 0., 0.), (1., 2., 0.), (3., 4., 1.), (5., 6., 2.)))
    samples = []
    # Includes the previous 150 ms boundary, a large gap and a trial change.
    for fid, (time, position) in enumerate(zip((65.6, 65.75, 70.8, 71.05), positions)):
        pose = np.eye(4)
        pose[:3, 3] = position
        samples.append((time, fid, 0, pose))
    groups = (0, 0, 1, 1)
    if components:
        write_velocity_component_colormaps(
            str(tmp_path), samples, samples, np.ones((4, 3)), groups=groups)
    else:
        write_error_colormap(str(tmp_path), samples, samples, np.ones(4),
                             'error.png', 'test', 'Error', 'mm', groups=groups)

    assert len(figures) == 1
    planar_axes = [axis for axis in figures[0].axes if 'XY' in axis.get_title()]
    assert len(planar_axes) == (3 if components else 1)
    for axis in planar_axes:
        # Every adjacent estimate pair is drawn, including the large time gap.
        np.testing.assert_allclose(
            axis.collections[0].get_segments(),
            np.stack((positions[:-1, :2], positions[1:, :2]), axis=1))
        # GT has the same complete line, with no inserted NaN breaks.
        np.testing.assert_allclose(axis.lines[0].get_xydata(), positions[:, :2])
