

# --------------------------------------------------------------------------- #
# Ancuti red-channel compensation
# --------------------------------------------------------------------------- #
def test_red_compensation_lifts_a_red_starved_image():
    """Water absorbs red first; this borrows structure from the green channel."""
    import numpy as np

    from netinspect.preprocess import compensate_red
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[..., 0] = 10        # red almost gone, as underwater
    img[..., 1] = 180       # green survives
    img[..., 2] = 160
    out = compensate_red(img)
    assert out[..., 0].mean() > img[..., 0].mean(), "red must be restored"
    assert np.array_equal(out[..., 1], img[..., 1]), "green must be untouched"
    assert np.array_equal(out[..., 2], img[..., 2]), "blue must be untouched"


def test_red_compensation_leaves_a_balanced_image_alone():
    """With red and green already equal the correction term is zero."""
    import numpy as np

    from netinspect.preprocess import compensate_red
    img = np.full((8, 8, 3), 120, dtype=np.uint8)
    assert np.abs(compensate_red(img).astype(int) - img.astype(int)).max() <= 1


def test_red_compensation_never_leaves_the_valid_range():
    import numpy as np

    from netinspect.preprocess import compensate_red
    rng = np.random.default_rng(0)
    out = compensate_red(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), alpha=3.0)
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255


def test_red_compensation_does_not_touch_already_bright_red():
    """The (1 - I_r) term concentrates the fix where red is actually depleted."""
    import numpy as np

    from netinspect.preprocess import compensate_red
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = 255       # red saturated
    img[..., 1] = 255
    before = img[..., 0].copy()
    assert np.abs(compensate_red(img)[..., 0].astype(int) - before.astype(int)).max() <= 1
