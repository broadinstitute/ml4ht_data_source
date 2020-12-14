import pytest

from ml4h.data.result import Result


def test_ok():
    result = Result.Data("cool")
    assert result.ok

    result = Result.Error("not cool")
    assert not result.ok


def test_bad_init():
    with pytest.raises(ValueError):
        Result("cool", "not cool")

    with pytest.raises(ValueError):
        Result(None, None)
