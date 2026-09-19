"""Lossless bitwise storage relative to immutable native checkpoint words.

This is XOR+deflate storage, not a model quantizer, sparse parameter selection,
numerical subtraction, mechanism extraction or transported activation delta.
Every native coordinate/parameter word is reconstructed exactly.
"""
import numpy as np


def encode_FP32(actual, original_BF16):
    actual = np.ascontiguousarray(actual, dtype=np.float32)
    original = np.asarray(original_BF16, dtype=np.uint16)
    assert actual.shape == original.shape and np.isfinite(actual).all()
    return np.bitwise_xor(actual.view(np.uint32), original.astype(np.uint32) << 16)


def decode_FP32(encoded, original_BF16):
    encoded = np.asarray(encoded, dtype=np.uint32)
    original = np.asarray(original_BF16, dtype=np.uint16)
    assert encoded.shape == original.shape
    return np.bitwise_xor(encoded, original.astype(np.uint32) << 16).view(np.float32)


def encode_BF16(actual_BF16, original_BF16):
    actual = np.asarray(actual_BF16, dtype=np.uint16)
    original = np.asarray(original_BF16, dtype=np.uint16)
    assert actual.shape == original.shape
    return np.bitwise_xor(actual, original)


def decode_BF16(encoded, original_BF16):
    return encode_BF16(encoded, original_BF16)
