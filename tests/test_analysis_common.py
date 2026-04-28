import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

# Stub the 'datasets' package so analysis.common can be imported without it.
# Use MagicMock so the stubs are callable and support attribute access,
# preventing them from breaking other tests that use the same stubs.
if "datasets" not in sys.modules:
    _datasets_stub = types.ModuleType("datasets")
    _datasets_stub.Audio = MagicMock(name="Audio")
    _datasets_stub.load_dataset = MagicMock(name="load_dataset")
    _datasets_stub.Dataset = MagicMock(name="Dataset")
    _datasets_stub.Features = MagicMock(name="Features")
    _datasets_stub.Value = MagicMock(name="Value")
    sys.modules["datasets"] = _datasets_stub

from analysis.common import has_valid_audio, init_result_entry, merge_model_result


class HasValidAudioTests(unittest.TestCase):
    def test_rejects_none(self):
        self.assertFalse(has_valid_audio(None))

    def test_rejects_empty_list(self):
        self.assertFalse(has_valid_audio([]))

    def test_rejects_empty_1d_array(self):
        self.assertFalse(has_valid_audio(np.array([])))

    def test_rejects_empty_2d_array(self):
        self.assertFalse(has_valid_audio(np.array([[]])))

    def test_rejects_scalar_ndarray(self):
        self.assertFalse(has_valid_audio(np.float32(0.0)))

    def test_rejects_dict(self):
        self.assertFalse(has_valid_audio({"array": [1, 2, 3]}))

    def test_accepts_valid_1d_array(self):
        self.assertTrue(has_valid_audio(np.array([0.1, 0.2, 0.3])))

    def test_accepts_valid_2d_array(self):
        self.assertTrue(has_valid_audio(np.array([[0.1, 0.2], [0.3, 0.4]])))

    def test_accepts_nonempty_list(self):
        self.assertTrue(has_valid_audio([0.1, 0.2]))


class InitResultEntryTests(unittest.TestCase):
    def test_has_required_fields(self):
        entry = init_result_entry(idx=0, audio_path="audio/001.wav", ref_text="hello")
        self.assertEqual(entry["id"], 0)
        self.assertEqual(entry["path"], "audio/001.wav")
        self.assertEqual(entry["ref_text"], "hello")
        self.assertEqual(entry["model_results"], {})
        self.assertEqual(entry["verification_status"], "pending")
        self.assertEqual(entry["score"], 0)

    def test_includes_audio_when_provided(self):
        audio = np.array([0.1, 0.2])
        entry = init_result_entry(idx=1, audio_path="a.wav", ref_text="t", audio_data=audio, sampling_rate=16000)
        self.assertIs(entry["audio_array"], audio)
        self.assertEqual(entry["sampling_rate"], 16000)

    def test_omits_audio_keys_when_not_provided(self):
        entry = init_result_entry(idx=2, audio_path="b.wav", ref_text="t")
        self.assertNotIn("audio_array", entry)
        self.assertNotIn("sampling_rate", entry)


class MergeModelResultTests(unittest.TestCase):
    def test_stores_model_result(self):
        entry = init_result_entry(0, "a.wav", "hello")
        merge_model_result(entry, "flash", "hello", 100, "hello", "hello", 90)
        self.assertIn("flash", entry["model_results"])
        self.assertEqual(entry["model_results"]["flash"]["score"], 100)

    def test_preserves_existing_models(self):
        entry = init_result_entry(0, "a.wav", "text")
        merge_model_result(entry, "flash", "text", 95, "text", "text", 90)
        merge_model_result(entry, "pro", "text!", 90, "text", "text!", 90)
        self.assertIn("flash", entry["model_results"])
        self.assertIn("pro", entry["model_results"])

    def test_picks_best_model(self):
        entry = init_result_entry(0, "a.wav", "text")
        merge_model_result(entry, "flash", "text", 80, "text", "text", 90)
        merge_model_result(entry, "pro", "text", 95, "text", "text", 90)
        self.assertEqual(entry["model_used"], "pro")
        self.assertEqual(entry["score"], 95)
        self.assertEqual(entry["verification_status"], "correct")

    def test_marks_incorrect_below_threshold(self):
        entry = init_result_entry(0, "a.wav", "text")
        merge_model_result(entry, "flash", "wrong", 50, "text", "wrong", 90)
        self.assertEqual(entry["verification_status"], "incorrect")


if __name__ == "__main__":
    unittest.main()
