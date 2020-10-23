import numpy as np
from src.utils.helper_som import is_prime, get_minimal_distance_factors, get_som_dimensions


def test_is_prime():
    inputs = [41, 42]
    assert is_prime(inputs[0]), f"{inputs[0]} should be a prime number."
    assert not is_prime(inputs[1]), f"{inputs[1]} should be a non-prime number."


def test_get_minimal_distance_factors():
    inputs = [12, 17]
    assert get_minimal_distance_factors(inputs[0]) == (3, 4), \
        f"Factors of {inputs[0]} whose distance is minimised should be 3 and 4."
    assert get_minimal_distance_factors(inputs[1]) == (3, 6), \
        f"Factors of {inputs[1]} whose distance is minimised should be 3 and 6."


def test_get_som_dimensions():
    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert get_som_dimensions(array) == (2, 4)
