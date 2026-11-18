"""Bounded observability for the required shared-protocol regression suite."""
import faulthandler
import sys
import unittest
faulthandler.enable()
faulthandler.dump_traceback_later(120,repeat=True)
suite=unittest.defaultTestLoader.discover('tests/dist_protocol')
result=unittest.TextTestRunner(verbosity=2).run(suite)
faulthandler.cancel_dump_traceback_later()
sys.exit(not result.wasSuccessful())
