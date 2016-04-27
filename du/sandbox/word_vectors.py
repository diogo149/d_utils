import numpy as np


UNKNOWN_TOKEN = "<UNKNOWN_TOKEN>"


def read_glove_format_file(path):
    """
    eg. glove.6B.50d.txt
    """
    word_to_vector = {}
    with open(path) as f:
        for line in f:
            row = line.split()
            word = row[0]
            vec = np.array(map(float, row[1:]))
            word_to_vector[word] = vec
    if UNKNOWN_TOKEN not in word_to_vector:
        # use mean vector as unknown token if not present
        unk_vec = np.mean(word_to_vector.values(), axis=0)
        word_to_vector[UNKNOWN_TOKEN] = unk_vec
    return word_to_vector


def write_glove_format_file(path, word_to_vector):
    with open(path, "w") as f:
        for word, vector in word_to_vector.items():
            # TODO parameterize float format
            f.write(" ".join([word] + ["%f" % val for val in vector]))
            f.write("\n")


class WordVectorWrapper(object):

    def __init__(self, word_to_vector):
        assert UNKNOWN_TOKEN in word_to_vector
        self.word_to_vector = word_to_vector

    @classmethod
    def from_glove_format_file(cls, path):
        return cls(word_to_vector=read_glove_format_file(path))

    def to_glove_format_file(self, path):
        write_glove_format_file(path, self.word_to_vector)

    def w2v(self, word):
        try:
            return self.word_to_vector[word]
        except KeyError:
            return self.word_to_vector[UNKNOWN_TOKEN]
