class Bar(object):

    def __init__(self, foo):
        self.foo = foo

    def run(self):
        self.bleh = 3
        raise ValueError(self.foo.MESSAGE)
