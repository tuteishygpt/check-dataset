from pathlib import Path
import unittest


UI_APP_PATH = Path(__file__).resolve().parents[1] / "ui" / "gradio_app.py"


class UiTextEncodingTests(unittest.TestCase):
    def test_gradio_ui_keeps_belarusian_copy_readable(self):
        source = UI_APP_PATH.read_text(encoding="utf-8")

        self.assertIn("Аналіз аўдыядатасетаў", source)
        self.assertIn("Увядзіце ваш HF Token", source)
        self.assertIn("Пачаць аналіз", source)

    def test_gradio_ui_does_not_contain_mojibake_markers(self):
        source = UI_APP_PATH.read_text(encoding="utf-8")

        for marker in ("Ã", "Ð", "Ñ"):
            self.assertNotIn(marker, source)


if __name__ == "__main__":
    unittest.main()
