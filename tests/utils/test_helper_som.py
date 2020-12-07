import pytest
import numpy as np

from src.utils.helper_som import is_prime, get_minimal_distance_factors, get_som_dimensions

# fixtures
numbers = {
    # note: need prime and non-prime to be same length
    'prime': [2, 3, 7, 17, 31, 101],
    'non-prime': [0, 1, 12, 27, 64, 171],
    'number': [12, 17],
    'number-factor': [(3, 4), (3, 6)]
}

simple_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])


# tests
@pytest.mark.parametrize("prime, non_prime", list(zip(numbers['prime'], numbers['non-prime'])))
def test_is_prime(prime, non_prime):
    assert is_prime(prime), f"{prime} should be a prime number."
    assert not is_prime(non_prime), f"{non_prime} should be a non-prime number."


@pytest.mark.parametrize("number, number_factors", list(zip(numbers['number'], numbers['number-factor'])))
def test_get_minimal_distance_factors(number, number_factors):
    assert get_minimal_distance_factors(number) == number_factors, \
        f"Factors of {number} whose distance is minimised should be {number_factors}."


def test_get_som_dimensions():
    assert get_som_dimensions(simple_array) == (2, 4)
