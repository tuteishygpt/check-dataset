import unittest
from unittest import mock

import numpy as np

from analysis import standard
from core.state import clear_global_results, get_global_results
import utils


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.features = {"audio": object()}

    def cast_column(self, name, feature):
        return self

    def select(self, indices):
        return FakeDataset([self.rows[idx] for idx in indices])

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class UtilsMemoryTests(unittest.TestCase):
    def test_load_hf_dataset_keeps_audio_lazy_by_default(self):
        dataset = FakeDataset(
            [
                {"audio": {"path": "a.wav", "bytes": b"aaa"}, "text": "one"},
                {"audio": {"path": "b.wav", "bytes": b"bbb"}, "text": "two"},
            ]
        )

        with mock.patch("utils.load_dataset", return_value=dataset), mock.patch(
            "utils.librosa.load", side_effect=AssertionError("audio decoded too early")
        ):
            loaded = utils.load_hf_dataset("demo/dataset", decode_audio=False)

        self.assertEqual(len(loaded), 2)
        self.assertIn("bytes", loaded[0]["audio"])
        self.assertNotIn("array", loaded[0]["audio"])

    def test_decode_audio_item_decodes_on_demand(self):
        item = {"audio": {"path": "a.wav", "bytes": b"aaa"}}

        with mock.patch("utils.librosa.load", return_value=(np.array([0.1, 0.2]), 16000)) as librosa_load:
            audio_array, sampling_rate, path = utils.decode_audio_item(item)

        librosa_load.assert_called_once()
        self.assertEqual(path, "a.wav")
        self.assertEqual(sampling_rate, 16000)
        self.assertEqual(audio_array.tolist(), [0.1, 0.2])


class AnalysisMemoryTests(unittest.TestCase):
    def setUp(self):
        clear_global_results()

    def tearDown(self):
        clear_global_results()

    def test_fresh_analysis_does_not_store_audio_arrays(self):
        dataset = FakeDataset(
            [
                {"audio": {"path": "a.wav", "bytes": b"aaa"}, "text": "alpha"},
                {"audio": {"path": "b.wav", "bytes": b"bbb"}, "text": "beta"},
            ]
        )
        gemini_tool = mock.Mock()
        gemini_tool.transcribe_audio.side_effect = ["alpha", "beta"]

        with mock.patch("analysis.common.get_cached_dataset", return_value=dataset), mock.patch(
            "analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", [])
        ), mock.patch("analysis.standard.save_results_csv"), mock.patch(
            "analysis.standard.utils.decode_audio_item",
            side_effect=[
                (np.array([0.1]), 16000, "a.wav"),
                (np.array([0.2]), 16000, "b.wav"),
            ],
        ):
            standard._run_fresh_analysis(
                gemini_tool=gemini_tool,
                model_name="gemini-2.5-flash-lite",
                dataset_name="demo/dataset",
                limit_files=0,
                similarity_threshold=90,
                gen_config={"temperature": 0.3},
                hf_token=None,
                progress=lambda *args, **kwargs: None,
            )

        results = get_global_results()
        self.assertEqual(len(results), 2)
        self.assertTrue(all("audio_array" not in result for result in results))
        self.assertTrue(all("sampling_rate" not in result for result in results))


if __name__ == "__main__":
    unittest.main()
