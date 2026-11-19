import numpy as np
import pytest

pytest.importorskip("tkinter")  # the module builds a Tk GUI; skip where Tk is missing
jlp = pytest.importorskip("journal_line_profiler")


def test_horizontal_line_on_ramp():
    img = np.tile(np.arange(50.0), (20, 1))  # value = x
    d, v = jlp.extract_line_profile(img, 5, 10, 45, 10)
    assert d[0] == 0 and d[-1] == pytest.approx(40)
    assert v == pytest.approx(5 + d, abs=1e-9)


def test_line_width_averages_across_the_line():
    img = np.zeros((21, 40))
    img[9:12, :] = 3.0  # 3-px bright stripe along y = 9..11
    _, v1 = jlp.extract_line_profile(img, 0, 10, 39, 10, line_width=1)
    _, v5 = jlp.extract_line_profile(img, 0, 10, 39, 10, line_width=5)
    assert v1 == pytest.approx(3.0)
    assert v5 == pytest.approx(3.0 * 3 / 5)


def test_rgb_channels():
    img = np.zeros((10, 10, 3))
    img[..., 0] = 1.0
    _, red = jlp.extract_line_profile(img, 0, 5, 9, 5, channel="red")
    _, lum = jlp.extract_line_profile(img, 0, 5, 9, 5, channel="luminance")
    assert red == pytest.approx(1.0)
    assert lum == pytest.approx(0.2989)


def test_diagonal_length():
    img = np.ones((30, 30))
    d, _ = jlp.extract_line_profile(img, 0, 0, 20, 20)
    assert d[-1] == pytest.approx(np.hypot(20, 20))
