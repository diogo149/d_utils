import string

_empty_trans = string.maketrans("", "")


def remove_punctuation(s):
    """
    using technique from
    http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    """
    return s.translate(_empty_trans, string.punctuation)


def remove_digits(s):
    """
    using technique from
    http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    """
    return s.translate(_empty_trans, string.digits)
