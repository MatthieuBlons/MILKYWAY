import logging
import time

def configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    verbose : bool, default=False
        Use informational logging when enabled and warning-level logging
        otherwise.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

class timetracker(object):
    """
    Stopwatch  timer
    """

    def __init__(self, name=None, verbose=False):
        self.name = name
        self.verbose = verbose
        self.TicToc = self.TicTocGenerator()

    def TicTocGenerator(self):  # add verbose arg
        # Generator that returns time differences
        ti = 0  # initial time
        tf = time.time()  # final time
        while True:
            ti = tf
            tf = time.time()
            yield tf - ti  # returns the time difference

    def toc(self, tempBool=True):
        tempTimeInterval = next(self.TicToc)
        if tempBool:
            if self.verbose:
                print("Elapsed time: %f seconds." % tempTimeInterval)
        return tempTimeInterval

    def tic(self):
        # Records a time in TicToc, marks the beginning of a time interval
        self.toc(False)