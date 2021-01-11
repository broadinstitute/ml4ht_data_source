import pytest

from ml4ht.data.transformation import Transformation, TransformationType


def test_is_augmentation():
    t = Transformation(TransformationType.AUGMENTATION, lambda x, y, z: 0)
    assert t.is_augmentation


def test_is_normalization():
    t = Transformation(TransformationType.NORMALIZATION, lambda x, y, z: 0)
    assert t.is_normalization


def test_is_filter():
    t = Transformation(TransformationType.FILTER, lambda x, y, z: 0)
    assert t.is_filter
