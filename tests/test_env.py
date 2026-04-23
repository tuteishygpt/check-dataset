import os
import tempfile
import unittest
from pathlib import Path

from core.env import configure_environment


class EnvironmentConfigTests(unittest.TestCase):
    def test_configure_environment_overrides_temp_and_resets_tempfile_cache(self):
        original_temp = os.environ.get("TEMP")
        original_tmp = os.environ.get("TMP")
        original_tempdir = tempfile.tempdir
        env_path = Path("test_env_override.env")

        try:
            env_path.write_text('TEMP="D:\\\\hf_tmp\\\\temp"\nTMP="D:\\\\hf_tmp\\\\temp"\n', encoding="utf-8")
            os.environ["TEMP"] = r"C:\Users\admin\AppData\Local\Temp"
            os.environ["TMP"] = r"C:\Users\admin\AppData\Local\Temp"
            tempfile.tempdir = r"C:\Users\admin\AppData\Local\Temp"

            configure_environment(str(env_path))

            self.assertEqual(os.environ["TEMP"], r"D:\hf_tmp\temp")
            self.assertEqual(os.environ["TMP"], r"D:\hf_tmp\temp")
            self.assertEqual(tempfile.gettempdir(), r"D:\hf_tmp\temp")
        finally:
            if env_path.exists():
                env_path.unlink()
            tempfile.tempdir = original_tempdir
            if original_temp is None:
                os.environ.pop("TEMP", None)
            else:
                os.environ["TEMP"] = original_temp
            if original_tmp is None:
                os.environ.pop("TMP", None)
            else:
                os.environ["TMP"] = original_tmp


if __name__ == "__main__":
    unittest.main()
