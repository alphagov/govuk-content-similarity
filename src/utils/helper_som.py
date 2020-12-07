from typing import Union
from math import sqrt
from math import ceil
from math import floor
import numpy as np


def is_prime(n: Union[int, float]) -> bool:
    """
    Checks if a number is prime.

    Reference:
    - https://stackoverflow.com/a/17377939/13416265

    :param n: Integer or float to check for prime property. If float, rounds to nearest int.
    :return: Boolean to state whether n is prime or not.
    """
    try:
        if isinstance(n, float):
            n = int(n)
            return is_prime(n=n)
        else:
            if n == 2:
                return True
            if n % 2 == 0 or n <= 1:
                return False

            root = int(sqrt(n)) + 1
            for divisor in range(3, root, 2):
                if n % divisor == 0:
                    return False
            return True
    except TypeError:
        print(f"Input '{n}' is not a integer nor float, please enter one!")


def get_minimal_distance_factors(n: Union[int, float]) -> (int, int):
    """
    Get the factors of a number which have the smallest difference between them.

    This is so we specify the gridsize for our SOM to be relatively balanced in height and width.

    Note:
    If have prime number, naively takes nearest even number.
    This is so we don't end up in situation where have 1xm gridsize.
    Might be better ways to do this.

    Intuition:
    The main logic here is using the square root of n as a starting point for computing factors.
    This is done for efficiency purposes so that we can reduce the set of numbers that can be factors of n.

    If we consider the integer n, then the set of possible factors is [1, ..., n].
    We can divide this set into two subsets, [1, ..., m] and [m+1, ..., n], where m < n.
    The method we can divide this set into two subsets can differ but taking the square root is probably optimal.
    This is because it minimises the size of the first subset, [1, ..., m], without omitting possible factors.
    Then we iterate back from m to m - 1, m - 2, ... until we get a divisor of n.

    If you consider the integer, 24, then the minimal distance factors are (4, 6).
    The square root of 24 is 4.898... which is rounded down to 4 and 24 / 4 = 6.

    If you consider the prime integer, 37, then the minimal distance factors are (1, 37).

    The square root of 37 is 6.082... which is rounded down to 6 and 37 % 6 > 0 (there is a remainder).
    Thus, you do 6 - 1 = 5 to check the next factor and 37 % 5 > 0.

    Repeat this process until you reach 6 - 1 - 1 - ... - 1 = 1 and 37 % 1 = 0.

    :param n: Integer or float we want to extract the closest factors from.
    :return: Tuple of integers representing factors of n whose distance from each other is minimised.
    """
    try:
        if isinstance(n, float) or is_prime(n):
            # gets next largest even number
            n = ceil(n / 2) * 2
            return get_minimal_distance_factors(n)
        else:
            root = floor(sqrt(n))
            while n % root > 0:
                root -= 1
            return int(root), int(n / root)
    except TypeError:
        print(f"Input '{n}' is not a integer nor float, please enter one!")


def get_som_dimensions(arr: np.array) -> (int, int):
    """
    Computes the number of neurons and how many they should make up each side in (x,y) dimensions for SOMs,
    where this is approximately the ratio of two largest eigenvalues of training data's covariance matrix. \n
    Reference:
    - https://python-data-science.readthedocs.io/en/latest/unsupervised.html

    Rule of thumb for setting grid is 5*sqrt(N) where N is sample size.

    Example must be transpose of our case:
    https://stats.stackexchange.com/questions/282288/som-grid-size-suggested-by-vesanto

    Note: This relies on division. In case where the divisor is 0, falls back to get_minimal_distance_factors().

    :param arr: numpy array of normalised vectors to go into SOMs.
    :return: Tuple of integers, (x, y), representing dimensions to input into SOM.
    """
    try:
        total_neurons = 5 * sqrt(arr.shape[0])
        # compute eigenvalues
        normal_cov = np.cov(arr.T)
        eigen_values = np.linalg.eigvals(normal_cov)
        # get two largest eigenvalues
        result = sorted([i.real for i in eigen_values])[-2:]
        # how do we deal with case when result[0] == 0?
        if result[0] == 0:
            print("Cannot divide by 0, computing minimal distance factors instead")
            return get_minimal_distance_factors(total_neurons)
        else:
            x = result[1] / result[0]
            y = total_neurons / x

            # round to nearest integer and convert to integer datatype
            return int(round(x, 0)), int(round(y, 0))
    except TypeError:
        print("Input '{}' is not a numpy array, please enter one!".format(arr))
