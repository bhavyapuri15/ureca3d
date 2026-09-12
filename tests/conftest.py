import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_addoption(parser):
    parser.addoption("--sol", action="store_true",
                     help="run the tests against solutions/patch.py instead of "
                          "your own implementations (sanity check of the tests)")


def pytest_configure(config):
    if config.getoption("--sol"):
        from solutions import patch
        patch.apply()
