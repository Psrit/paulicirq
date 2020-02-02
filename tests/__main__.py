import os
import sys
import unittest


def main():
    suite = unittest.defaultTestLoader.discover(
        os.path.dirname(os.path.abspath(__file__)), "*test.py"
    )

    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    runner = unittest.TextTestRunner(stream=devnull, verbosity=2)
    result = runner.run(suite)
    sys.stdout = sys.__stdout__

    print("Ran {} test{}."
          .format(result.testsRun, result.testsRun != 1 and "s" or ""))
    if not result.wasSuccessful():
        print("FAILED")
        failed, errored = len(result.failures), len(result.errors)
        if failed:
            print("failures=%d" % failed)
        if errored:
            print("errors=%d" % errored)
    else:
        print("OK")


if __name__ == "__main__":
    main()
