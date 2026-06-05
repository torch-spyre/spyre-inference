import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))

pytest_plugins = ("spyre_testing_plugin.pytest_plugin",)
