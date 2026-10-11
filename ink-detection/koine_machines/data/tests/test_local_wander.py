import numpy as np

from koine_machines.data.ink_dataset import (
    _apply_local_wander,
    _local_wander_settings,
    _smooth_field,
)


def _ramp_volume(depth=21, size=32):
    """Each layer z holds the value 10*z, so the sampled depth can be read back from the output."""
    return (np.arange(depth, dtype=np.float32)[:, None, None] * 10.0) * np.ones((depth, size, size), np.float32)


def test_zero_amplitude_equals_whole_window_shift():
    full = _ramp_volume()
    for offset in (-2, 0, 2):
        out = _apply_local_wander(full, window_depth=17, window_offset=offset, max_amplitude=0.0,
                                  corr_range=(64, 256), rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out, full[2 + offset: 2 + offset + 17])


def test_wander_stays_inside_real_layers_and_varies_in_plane():
    full = _ramp_volume()
    for seed in range(20):
        out = _apply_local_wander(full, window_depth=17, window_offset=2, max_amplitude=2.0,
                                  corr_range=(8, 16), rng=np.random.default_rng(seed))
        depth_read = out / 10.0                       # first output layer sits at 2 + shift
        assert depth_read.min() >= 0.0 and depth_read.max() <= 20.0
        assert (depth_read[0] >= 2.0 - 2.0 - 1e-5).all() and (depth_read[0] <= 2.0 + 2.0 + 1e-5).all()
    out = _apply_local_wander(full, window_depth=17, window_offset=0, max_amplitude=2.0,
                              corr_range=(8, 16), rng=np.random.default_rng(3))
    assert np.ptp(out[0]) > 0.0                        # the start depth differs across the patch


def test_integer_input_keeps_dtype():
    full = (_ramp_volume() % 256).astype(np.uint8)
    out = _apply_local_wander(full, window_depth=17, window_offset=0, max_amplitude=1.0,
                              corr_range=(64, 256), rng=np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == (17, 32, 32)


def test_smooth_field_peak_and_smoothness():
    field = _smooth_field((128, 128), corr_px=64, rng=np.random.default_rng(0))
    assert abs(np.abs(field).max() - 1.0) < 1e-6
    assert np.abs(np.diff(field, axis=0)).max() < 0.2   # no pixel-scale noise


def test_settings_only_for_augmented_training_patches():
    cfg = {"flat_z_local_wander": {"enabled": True, "max_amplitude": 2.0, "corr_px": [64, 256]}}
    assert _local_wander_settings(cfg, do_augmentations=True, is_validation=False) == (2.0, (64.0, 256.0))
    assert _local_wander_settings(cfg, do_augmentations=False, is_validation=False) is None
    assert _local_wander_settings(cfg, do_augmentations=True, is_validation=True) is None
    assert _local_wander_settings({}, do_augmentations=True, is_validation=False) is None
