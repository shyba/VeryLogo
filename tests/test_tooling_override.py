import os
import unittest

from stc.tooling import require_tool


class TestToolingOverride(unittest.TestCase):
    def test_env_override(self) -> None:
        key = "STC_FOO"
        old = os.environ.get(key)
        try:
            os.environ[key] = "/tmp/foo"
            self.assertEqual(require_tool("foo"), "/tmp/foo")
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
