import math
import unittest

from ui.audio import array_to_b64_audio, get_audio_for_row


class UiAudioTests(unittest.TestCase):
    def test_array_to_b64_audio_returns_placeholder_for_nan(self):
        html = array_to_b64_audio(float("nan"), 16000)

        self.assertIn("Аўдыя недаступна", html)

    def test_get_audio_for_row_returns_none_for_nan_audio(self):
        row = {
            "audio_array": float("nan"),
            "sampling_rate": 16000,
        }

        result = get_audio_for_row([row], 0)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
