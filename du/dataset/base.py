import abc


# ############################### base dataset ###############################


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _open(self):
        """
        returns a datamap generator and performs setup (if any)
        """

    def _close(self):
        """
        performs cleanup (if any)
        """

    def opened(self):
        if not hasattr(self, "opened_"):
            self.opened_ = False
        return self.opened_

    def open(self):
        if self.opened():
            raise Exception("Dataset has already been opened")
        self.opened_ = True
        return self._open()

    def close(self):
        if not self.opened():
            raise Exception("Dataset can't be closed before opened")
        self.opened_ = False
        self._close()

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, tb):
        self.close()
        # don't supress any exception
        return False

    def __getstate__(self):
        assert not self.opened()
        return self.__dict__.copy()

    def __setstate__(self, dict):
        """
        default set state... implementing this so that child classes can call
        super
        """
        self.__dict__ = dict
