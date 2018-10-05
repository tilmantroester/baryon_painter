class Painter:
    """Abstract base class for a baryon painter.

    This class should be sub-classed and the methods ``load_state`` and 
    ``paint`` implemented.
    """

    def __init__(self):
        raise NotImplementedError("This is an abstract base class.")

    def load_state_from_file(self, filename):
        raise NotImplementedError("This is an abstract base class.")

    def paint(self, input):
        raise NotImplementedError("This is an abstract base class.")