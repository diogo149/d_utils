import du
from du._test_utils import equal


def test_remove_punctuation():
    equal("abcdefg",
          du.string_utils.remove_punctuation("a!b,,,.cd`'\"`efg."))


def test_remove_digits():
    equal("abcdefg",
          du.string_utils.remove_digits("1a2b3cd5e6f7g0"))
