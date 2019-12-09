import string


def remove_punctuation(s):
    """
    using technique from
    http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python

    note: there's an updated python 3 version for handling unicode
    https://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings/21635971#21635971
    """
    return s.translate({ord(c): None for c in string.punctuation})


def remove_digits(s):
    """
    using technique from
    http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    """
    return s.translate({ord(c): None for c in string.digits})
