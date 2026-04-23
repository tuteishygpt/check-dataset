import unittest
from unittest import mock

import numpy as np

from analysis.import_export import create_verified_dataset
from core.state import clear_global_results, set_global_results


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.features = {}
        self.info = mock.Mock()

    def __len__(self):
        return len(self.rows)

    def push_to_hub(self, *args, **kwargs):
        return None


class VerifiedDatasetExportTests(unittest.TestCase):
    def tearDown(self):
        clear_global_results()

    @mock.patch("analysis.import_export.HfApi")
    @mock.patch("analysis.import_export.login")
    @mock.patch("analysis.import_export.Dataset.from_generator")
    @mock.patch("analysis.import_export.utils.load_hf_dataset")
    def test_create_verified_dataset_recovers_audio_by_source_idx(
        self,
        load_hf_dataset_mock,
        from_generator_mock,
        login_mock,
        hf_api_cls,
    ):
        set_global_results(
            [
                {
                    "id": 0,
                    "source_idx": 0,
                    "path": "renamed-a.wav",
                    "ref_text": "alpha",
                    "verification_status": "correct",
                },
                {
                    "id": 1,
                    "source_idx": 2,
                    "path": "renamed-c.wav",
                    "ref_text": "charlie",
                    "verification_status": "correct",
                },
            ]
        )
        source_dataset = [
            {"audio": {"path": "source-a.wav", "bytes": b"raw-a"}},
            {"audio": {"path": "source-b.wav", "bytes": b"raw-b"}},
            {"audio": {"path": "source-c.wav", "bytes": b"raw-c"}},
        ]
        load_hf_dataset_mock.return_value = source_dataset

        def fake_decode_audio_item(item):
            path = item["audio"]["path"]
            return np.array([0.1, 0.2], dtype=np.float32), 16000, path

        generated_rows = []

        def consume_generator(gen_fn, features=None):
            generated_rows.extend(list(gen_fn()))
            return FakeDataset(generated_rows)

        hf_api_cls.return_value.whoami.return_value = {"name": "tester"}
        from_generator_mock.side_effect = consume_generator

        with mock.patch("analysis.import_export.utils.decode_audio_item", side_effect=fake_decode_audio_item):
            result = create_verified_dataset(
                hf_token="hf_test",
                dataset_name="owner/source",
                progress=lambda *args, **kwargs: None,
            )

        self.assertIn("tester/sourceChecked", result)
        self.assertEqual([row["text"] for row in generated_rows], ["alpha", "charlie"])
        self.assertEqual([row["original_path"] for row in generated_rows], ["renamed-a.wav", "renamed-c.wav"])
        load_hf_dataset_mock.assert_called_once_with(
            "owner/source",
            limit=3,
            hf_token="hf_test",
            decode_audio=False,
        )

    @mock.patch("analysis.import_export.HfApi")
    @mock.patch("analysis.import_export.login")
    @mock.patch("analysis.import_export.Dataset.from_generator")
    @mock.patch("analysis.import_export.utils.load_hf_dataset")
    def test_create_verified_dataset_skips_duplicate_verified_records(
        self,
        load_hf_dataset_mock,
        from_generator_mock,
        login_mock,
        hf_api_cls,
    ):
        set_global_results(
            [
                {
                    "id": 0,
                    "source_idx": 0,
                    "path": "first-a.wav",
                    "ref_text": "alpha",
                    "verification_status": "correct",
                },
                {
                    "id": 1,
                    "source_idx": 0,
                    "path": "duplicate-a.wav",
                    "ref_text": "duplicate alpha",
                    "verification_status": "correct",
                },
                {
                    "id": 2,
                    "source_idx": 1,
                    "path": "first-b.wav",
                    "ref_text": "bravo",
                    "verification_status": "correct",
                },
            ]
        )
        load_hf_dataset_mock.return_value = [
            {"audio": {"path": "source-a.wav", "bytes": b"raw-a"}},
            {"audio": {"path": "source-b.wav", "bytes": b"raw-b"}},
        ]

        def fake_decode_audio_item(item):
            return np.array([0.1, 0.2], dtype=np.float32), 16000, item["audio"]["path"]

        generated_rows = []

        def consume_generator(gen_fn, features=None):
            generated_rows.extend(list(gen_fn()))
            return FakeDataset(generated_rows)

        hf_api_cls.return_value.whoami.return_value = {"name": "tester"}
        from_generator_mock.side_effect = consume_generator

        with mock.patch("analysis.import_export.utils.decode_audio_item", side_effect=fake_decode_audio_item):
            result = create_verified_dataset(
                hf_token="hf_test",
                dataset_name="owner/source",
                progress=lambda *args, **kwargs: None,
            )

        self.assertEqual([row["text"] for row in generated_rows], ["alpha", "bravo"])
        self.assertIn("2 unique records", result)
        self.assertIn("1 duplicate skipped", result)


if __name__ == "__main__":
    unittest.main()
