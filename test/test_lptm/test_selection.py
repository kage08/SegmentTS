import numpy as np

from lptm.selection import pad_times


def test_pad_times_basic():
    scores = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([1, 2], dtype=np.int32)

    expected_output = np.array(
        [
            [
                [1.0, -np.inf, -np.inf],
                [-np.inf, -np.inf, -np.inf],
                [-np.inf, -np.inf, -np.inf],
            ],
            [[9.0, 8.0, -np.inf], [6.0, 5.0, -np.inf], [-np.inf, -np.inf, -np.inf]],
        ],
        dtype=np.float32,
    )

    output = pad_times(scores, times)
    np.testing.assert_array_equal(output, expected_output)


def test_pad_times_all_times_zero():
    scores = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32
    )
    times = np.array([0], dtype=np.int32)

    expected_output = np.array(
        [
            [
                [-np.inf, -np.inf, -np.inf],
                [-np.inf, -np.inf, -np.inf],
                [-np.inf, -np.inf, -np.inf],
            ]
        ],
        dtype=np.float32,
    )

    output = pad_times(scores, times)
    np.testing.assert_array_equal(output, expected_output)


def test_pad_times_no_padding_needed():
    scores = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=np.float32
    )
    times = np.array([3], dtype=np.int32)

    expected_output = scores

    output = pad_times(scores, times)
    np.testing.assert_array_equal(output, expected_output)
