from .. import joblib_utils

from . import base

# ############################### constructors ###############################


class FromListDataset(base.Dataset):

    def __init__(self, datamaps):
        """
        dataset constructor that takes in a list of datamaps
        """
        self.datamaps = datamaps

    def _open(self):
        for datamap in self.datamaps:
            yield datamap


class FromGeneratorDataset(base.Dataset):

    def __init__(self, generator):
        """
        dataset constructor that takes in a generator of datamaps.

        Note: this should be the instantiated generator. So for a function:

        def generate_data():
            while index < len(data):
                yield data[index].load()
                index += 1

        this constructor will receive generate_data(), not generate_data, as
        the parameter.

        To pass in the function itself, use the FromGeneratorFnDataset method.
        """
        self.generator = generator

    def _open(self):
        return self.generator


class FromGeneratorFnDataset(base.Dataset):

    def __init__(self, generator_fn):
        """
        dataset constructor that takes in a 0-arg function that returns a
        generator of datamaps

        Note: this should be the uninstantiated generator. So for a function:

        def generate_data():
            while index < len(data):
                yield data[index].load()
                index += 1

        this constructor will receive generate_data, not generate_data(), as
        the parameter.

        To pass in the instantiated generator itself, use the
        FromGeneratorDataset method.

        use case: this is for multiprocessing - generators are not serializable
        but top-level functions are, so this takes in a function that returns
        a generator as a hack around serializable generators
        """
        self.generator_fn = generator_fn

    def _open(self):
        return self.generator_fn()


class FromJoblibDirectoryDataset(base.Dataset):

    def __init__(self, dirname, mmap_mode=None):
        """
        dataset constructor that takes in a path to a directory with joblib
        serialized code of a list of datamaps
        """
        self.dirname = dirname
        self.mmap_mode = mmap_mode

    def _open(self):
        return joblib_utils.load_dir(self.dirname,
                                     mmap_mode=self.mmap_mode)


class FromJoblibCachedDataset(base.Dataset):

    def __init__(self, dirname, mmap_mode=None):
        """
        dataset constructor that takes in a path to a directory with joblib
        serialized code from a JoblibCachedDataset
        """
        self.dirname = dirname
        self.mmap_mode = mmap_mode

    def _open(self):
        return joblib_utils.load_dir(self.dirname,
                                     mmap_mode=self.mmap_mode)["data"]
