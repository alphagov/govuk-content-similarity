import pytest
import pandas as pd


@pytest.fixture
def text():
    return pd.DataFrame(data={'text_dirty': ["According to all known laws of aviation, there is no way a bee should be able to fly.",  # noqa: E501
                                             "Its wings are too small to get its fat little body off the ground.",
                                             "The bee, of course, flies anyway because bees don't care what humans think is impossible.",  # noqa:E501
                                             "Yellow, black. Yellow, black."],
                              'text_clean': ["accord know law aviation , way bee able fly .",
                                             "wing small fat little body ground .",
                                             "bee , course , fly bee care human think impossible .",
                                             "yellow , black . yellow , black ."]})
