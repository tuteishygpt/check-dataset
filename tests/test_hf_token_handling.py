import unittest
from unittest import mock

from hf_asr import get_hf_asr_client
import utils


class HuggingFaceTokenHandlingTests(unittest.TestCase):
    def test_hf_asr_client_treats_blank_token_as_missing(self):
        client = get_hf_asr_client("SeamlessM4T-v2 (HF)", hf_token="")

        self.assertIsNone(client.hf_token)

    @mock.patch("utils.load_dataset")
    def test_load_hf_dataset_treats_blank_token_as_missing(self, load_dataset):
        dataset = mock.Mock()
        dataset.features = {}
        dataset.__len__ = mock.Mock(return_value=0)
        dataset.__iter__ = mock.Mock(return_value=iter([]))
        dataset.select.return_value = dataset
        load_dataset.return_value = dataset

        utils.load_hf_dataset("demo/dataset", hf_token="")

        self.assertEqual(load_dataset.call_args.kwargs["token"], None)


if __name__ == "__main__":
    unittest.main()
