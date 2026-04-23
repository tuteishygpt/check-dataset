import io
import unittest

from core.console import safe_print


class Cp1252Console(io.StringIO):
    encoding = "cp1252"


class SafePrintTests(unittest.TestCase):
    def test_safe_print_replaces_unencodable_characters_for_cp1252(self):
        stream = Cp1252Console()

        safe_print("✅ Done", file=stream)

        self.assertEqual(stream.getvalue(), "? Done\n")


if __name__ == "__main__":
    unittest.main()
