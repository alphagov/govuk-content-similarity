from math import sqrt
from math import ceil
from math import floor
import numpy as np


def is_prime(n):
    """
    Checks if a number is prime.
    https://stackoverflow.com/a/17377939/13416265

    :param n: Number to check if prime
    :return: Boolean to state whether n is prime or not
    """
    try:
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


def get_minimal_distance_factors(n):
    """
    Get the factors of a number which have the smallest difference between them.
    This is so we specify the gridsize for our SOM to be relatively balanced in height and width.
    Note: If have prime number, naively takes nearest even number.
            This is so we don't end up in situation where have 1xm gridsize.
            Might be better ways to do this.

    :param n: Integer or float we want to extract the closest factors from.
    :return: The factors of n whose distance from each other is minimised.
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


def get_som_dimensions(arr):
    """
    Computes the number of neurons and how many they should make up each side in (x,y) dimensions for SOMs,
    where this is approximately the ratio of two largest eigenvalues of training data's covariance matrix.
    https://python-data-science.readthedocs.io/en/latest/unsupervised.html
    Rule of thumb for setting grid is 5*sqrt(N) where N is sample size.
    Example must be transpose of our case:
    https://stats.stackexchange.com/questions/282288/som-grid-size-suggested-by-vesanto

    Note: This relies on division. In case where the divisor is 0, falls back to get_minimal_distance_factors().

    :param arr: numpy array of normalised vectors to go into SOMs.
    :return: The (x, y) dimensions to input into SOM.
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
