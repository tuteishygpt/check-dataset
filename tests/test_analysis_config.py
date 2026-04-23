import unittest
from unittest.mock import Mock, patch

import gradio as gr

from core.state import clear_global_results, set_global_results
from ui.gradio_app import _run_smart_analysis_with_mode, update_model_controls
from analysis.smart import build_smart_generation_config
from analysis.standard import (
    ANALYSIS_SCOPE_ALL,
    ANALYSIS_SCOPE_PENDING,
    ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
    ANALYSIS_SCOPE_PROBLEMATIC,
    _run_hf_recheck_analysis,
    _run_recheck_analysis,
    _run_vertex_batch_analysis,
    build_model_generation_config,
    get_analysis_target_indices,
    normalize_execution_mode,
    normalize_analysis_scope,
    run_analysis,
)


class AnalysisConfigTests(unittest.TestCase):
    def tearDown(self):
        clear_global_results()

    def test_normalize_execution_mode_defaults_to_direct(self):
        self.assertEqual(normalize_execution_mode(None), "direct")

    def test_normalize_execution_mode_preserves_supported_modes(self):
        self.assertEqual(normalize_execution_mode("direct"), "direct")
        self.assertEqual(normalize_execution_mode("flex"), "flex")
        self.assertEqual(normalize_execution_mode("batch"), "batch")

    def test_normalize_execution_mode_uses_legacy_flex_flag(self):
        self.assertEqual(normalize_execution_mode(None, flex_mode=True), "flex")

    def test_normalize_execution_mode_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            normalize_execution_mode("turbo")

    def test_normalize_analysis_scope_defaults_to_all(self):
        self.assertEqual(normalize_analysis_scope(None), ANALYSIS_SCOPE_ALL)

    def test_normalize_analysis_scope_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            normalize_analysis_scope("custom")

    def test_get_analysis_target_indices_returns_all_records(self):
        results = [
            {"verification_status": "correct", "score": 99},
            {"verification_status": "incorrect", "score": 80},
            {"verification_status": "pending", "score": 0},
        ]

        self.assertEqual(
            get_analysis_target_indices(results, ANALYSIS_SCOPE_ALL, similarity_threshold=95),
            [0, 1, 2],
        )

    def test_get_analysis_target_indices_returns_only_problematic_records(self):
        results = [
            {"verification_status": "correct", "score": 99},
            {"verification_status": "incorrect", "score": 80},
            {"verification_status": "pending", "score": 0},
            {"verification_status": "correct", "score": 40},
        ]

        self.assertEqual(
            get_analysis_target_indices(results, ANALYSIS_SCOPE_PROBLEMATIC, similarity_threshold=95),
            [1, 2],
        )

    def test_get_analysis_target_indices_returns_only_pending_records(self):
        results = [
            {"verification_status": "correct", "score": 99},
            {"verification_status": "incorrect", "score": 80},
            {"verification_status": "pending", "score": 0},
            {"verification_status": "pending", "score": 30},
        ]

        self.assertEqual(
            get_analysis_target_indices(results, ANALYSIS_SCOPE_PENDING, similarity_threshold=95),
            [2, 3],
        )

    def test_get_analysis_target_indices_returns_problematic_and_pending_without_duplicates(self):
        results = [
            {"verification_status": "correct", "score": 99},
            {"verification_status": "incorrect", "score": 80},
            {"verification_status": "pending", "score": 0},
            {"verification_status": "pending", "score": 96},
        ]

        self.assertEqual(
            get_analysis_target_indices(
                results,
                ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
                similarity_threshold=95,
            ),
            [1, 2, 3],
        )

    def test_get_analysis_target_indices_applies_limit_after_filtering(self):
        results = [
            {"verification_status": "pending", "score": 0},
            {"verification_status": "incorrect", "score": 80},
            {"verification_status": "pending", "score": 30},
        ]

        self.assertEqual(
            get_analysis_target_indices(
                results,
                ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
                similarity_threshold=95,
                limit_files=2,
            ),
            [0, 1],
        )

    def test_standard_model_config_rejects_unsupported_flex_model(self):
        with self.assertRaises(ValueError):
            build_model_generation_config(
                model_name="gemini-1.5-pro",
                temperature=0.3,
                thinking_budget=0,
                flex_mode=True,
                location="global",
            )

    def test_standard_model_config_enables_thinking_for_thinking_models(self):
        config = build_model_generation_config(
            model_name="gemini-2.5-thinking",
            temperature=0.4,
            thinking_budget=1024,
            flex_mode=False,
            location="global",
        )

        self.assertEqual(config["thinking_config"]["budget_tokens"], 1024)

    def test_smart_generation_config_rejects_non_global_flex(self):
        with self.assertRaises(ValueError):
            build_smart_generation_config(temperature=0.2, flex_mode=True, location="us-central1")

    def test_smart_generation_config_enables_flex_on_global(self):
        config = build_smart_generation_config(temperature=0.2, flex_mode=True, location="global")
        self.assertEqual(
            config["http_options"]["headers"]["X-Vertex-AI-LLM-Shared-Request-Type"],
            "flex",
        )

    def test_smart_analysis_rejects_batch_mode(self):
        with self.assertRaises(gr.Error) as error:
            _run_smart_analysis_with_mode(
                dataset_name="demo/dataset",
                limit_files=1,
                temperature=0.2,
                thinking_budget=0,
                similarity_threshold=95,
                execution_mode="batch",
                analysis_scope=ANALYSIS_SCOPE_PENDING,
                recheck_problematic=False,
                hf_token=None,
            )

        self.assertIn("Batch mode", str(error.exception))

    @patch("ui.gradio_app.run_smart_analysis", return_value=("stats", "flagged", "table"))
    def test_smart_analysis_with_mode_forwards_analysis_scope(
        self,
        run_smart_analysis_mock,
    ):
        outputs = _run_smart_analysis_with_mode(
            dataset_name="demo/dataset",
            limit_files=1,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="direct",
            analysis_scope=ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
            recheck_problematic=False,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        self.assertEqual(
            run_smart_analysis_mock.call_args.kwargs["analysis_scope"],
            ANALYSIS_SCOPE_PENDING_OR_PROBLEMATIC,
        )

    def test_model_controls_reset_batch_for_unsupported_model(self):
        thinking_update, execution_mode_update = update_model_controls(
            "meta/llama-4-maverick",
            "batch",
        )

        self.assertFalse(thinking_update["visible"])
        self.assertEqual(execution_mode_update["value"], "direct")
        self.assertEqual(
            execution_mode_update["choices"],
            [("Direct Request", "direct"), ("Flex Mode", "flex")],
        )

    def test_model_controls_keep_execution_mode_for_hf_model(self):
        thinking_update, execution_mode_update = update_model_controls(
            "SeamlessM4T-v2 (HF)",
            "batch",
        )

        self.assertFalse(thinking_update["visible"])
        self.assertEqual(execution_mode_update["value"], "direct")
        self.assertEqual(execution_mode_update["choices"], [("Direct Request", "direct")])

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard.gr.Warning")
    @patch("analysis.standard._run_recheck_analysis", return_value=("direct", "flagged", "table"))
    @patch("analysis.standard._run_vertex_batch_analysis", return_value=("batch", "flagged", "table"))
    @patch("analysis.standard.validate_batch_inference")
    @patch("analysis.standard.GeminiIntegrator")
    def test_standard_analysis_uses_vertex_batch_for_problematic_scope_in_batch_mode(
        self,
        gemini_integrator_cls,
        validate_batch_inference_mock,
        run_vertex_batch_analysis_mock,
        recheck_analysis_mock,
        warning_mock,
        save_results_csv_mock,
    ):
        gemini_integrator_cls.return_value.location = "global"
        validate_batch_inference_mock.return_value = "gs://demo/prefix"

        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="gemini-2.5-flash-lite",
            limit_files=1,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="batch",
            analysis_scope=ANALYSIS_SCOPE_PROBLEMATIC,
            hf_token=None,
        )

        self.assertEqual(outputs, ("batch", "flagged", "table"))
        run_vertex_batch_analysis_mock.assert_called_once()
        self.assertEqual(
            run_vertex_batch_analysis_mock.call_args.kwargs["analysis_scope"],
            ANALYSIS_SCOPE_PROBLEMATIC,
        )
        recheck_analysis_mock.assert_not_called()
        warning_mock.assert_not_called()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.GeminiIntegrator")
    def test_standard_analysis_rejects_unknown_recognition_model(
        self,
        gemini_integrator_cls,
    ):
        with self.assertRaises(gr.Error):
            run_analysis(
                dataset_name="demo/dataset",
                model_name="meta/llama-4-maverick",
                limit_files=1,
                temperature=0.2,
                thinking_budget=0,
                similarity_threshold=95,
                execution_mode="batch",
                recheck_problematic=False,
                hf_token=None,
            )

        gemini_integrator_cls.assert_not_called()

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard._run_hf_asr_analysis", return_value=("stats", "flagged", "table"))
    def test_standard_analysis_ignores_vertex_mode_for_hf_model(
        self,
        run_hf_asr_analysis_mock,
        save_results_csv_mock,
    ):
        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="SeamlessM4T-v2 (HF)",
            limit_files=1,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="turbo",
            recheck_problematic=False,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        run_hf_asr_analysis_mock.assert_called_once()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard.GeminiIntegrator")
    @patch("analysis.standard._run_hf_asr_analysis", return_value=("stats", "flagged", "table"))
    def test_standard_analysis_does_not_initialize_vertex_for_hf_model(
        self,
        run_hf_asr_analysis_mock,
        gemini_integrator_cls,
        save_results_csv_mock,
    ):
        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="SeamlessM4T-v2 (HF)",
            limit_files=1,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="batch",
            recheck_problematic=False,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        run_hf_asr_analysis_mock.assert_called_once()
        gemini_integrator_cls.assert_not_called()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard._run_recheck_analysis", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.GeminiIntegrator")
    def test_standard_analysis_uses_recheck_path_in_direct_mode(
        self,
        gemini_integrator_cls,
        recheck_analysis_mock,
        save_results_csv_mock,
    ):
        gemini_integrator_cls.return_value.location = "global"

        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="gemini-2.5-flash-lite",
            limit_files=2,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="direct",
            recheck_problematic=True,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        recheck_analysis_mock.assert_called_once()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard._run_recheck_analysis", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.GeminiIntegrator")
    def test_standard_analysis_uses_recheck_path_in_flex_mode(
        self,
        gemini_integrator_cls,
        recheck_analysis_mock,
        save_results_csv_mock,
    ):
        gemini_integrator_cls.return_value.location = "global"

        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="gemini-3.1-flash-lite-preview",
            limit_files=2,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="flex",
            recheck_problematic=True,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        recheck_analysis_mock.assert_called_once()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard.gr.Warning")
    @patch("analysis.standard._run_vertex_batch_analysis", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.validate_batch_inference")
    @patch("analysis.standard.GeminiIntegrator")
    def test_standard_analysis_uses_vertex_batch_for_pending_scope_in_batch_mode(
        self,
        gemini_integrator_cls,
        validate_batch_inference_mock,
        run_vertex_batch_analysis_mock,
        warning_mock,
        save_results_csv_mock,
    ):
        gemini_integrator_cls.return_value.location = "global"
        validate_batch_inference_mock.return_value = "gs://demo/prefix"

        outputs = run_analysis(
            dataset_name="demo/dataset",
            model_name="gemini-2.5-flash-lite",
            limit_files=2,
            temperature=0.2,
            thinking_budget=0,
            similarity_threshold=95,
            execution_mode="batch",
            analysis_scope=ANALYSIS_SCOPE_PENDING,
            hf_token=None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        run_vertex_batch_analysis_mock.assert_called_once()
        warning_mock.assert_not_called()
        save_results_csv_mock.assert_called()

    @patch("analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.utils.calculate_similarity", side_effect=[(96, "ref1", "hyp1"), (94, "ref2", "hyp2")])
    @patch("analysis.standard.get_cached_dataset")
    @patch("builtins.print")
    def test_recheck_analysis_processes_only_problematic_files(
        self,
        print_mock,
        get_cached_dataset_mock,
        calculate_similarity_mock,
        generate_dashboard_outputs_mock,
    ):
        set_global_results([
            {
                "id": 0,
                "path": "ok.wav",
                "ref_text": "ok",
                "score": 99,
                "verification_status": "correct",
                "audio_array": [0.1],
                "sampling_rate": 16000,
            },
            {
                "id": 1,
                "path": "bad1.wav",
                "ref_text": "bad1",
                "score": 80,
                "verification_status": "incorrect",
                "audio_array": [0.2],
                "sampling_rate": 16000,
            },
            {
                "id": 2,
                "path": "pending.wav",
                "ref_text": "pending",
                "score": 0,
                "verification_status": "pending",
                "audio_array": [0.3],
                "sampling_rate": 16000,
            },
            {
                "id": 3,
                "path": "verified.wav",
                "ref_text": "verified",
                "score": 50,
                "verification_status": "correct",
                "audio_array": [0.4],
                "sampling_rate": 16000,
            },
        ])
        get_cached_dataset_mock.return_value = [
            {"audio": {"path": "ok.wav", "array": [0.1], "sampling_rate": 16000}},
            {"audio": {"path": "bad1.wav", "array": [0.2], "sampling_rate": 16000}},
            {"audio": {"path": "pending.wav", "array": [0.3], "sampling_rate": 16000}},
            {"audio": {"path": "verified.wav", "array": [0.4], "sampling_rate": 16000}},
        ]
        gemini_tool = Mock()
        gemini_tool.transcribe_audio.side_effect = ["hyp-one", "hyp-two"]

        outputs = _run_recheck_analysis(
            gemini_tool=gemini_tool,
            model_name="gemini-2.5-flash-lite",
            dataset_name="demo/dataset",
            limit_files=0,
            similarity_threshold=95,
            gen_config={"temperature": 0.2},
            hf_token=None,
            progress=lambda *args, **kwargs: None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        self.assertEqual(gemini_tool.transcribe_audio.call_count, 2)
        transcribed_paths = [
            call.args[1]
            for call in gemini_tool.transcribe_audio.call_args_list
        ]
        self.assertEqual(transcribed_paths, [[0.2], [0.3]])
        self.assertEqual(calculate_similarity_mock.call_count, 2)
        generate_dashboard_outputs_mock.assert_called_once_with(95)

    @patch("analysis.standard.save_results_csv")
    @patch("analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.utils.calculate_similarity", return_value=(97, "ref", "hyp"))
    @patch("analysis.standard.utils.decode_audio_item", return_value=([0.9], 16000, "bad.wav"))
    @patch("analysis.standard.get_cached_dataset")
    def test_hf_recheck_decodes_raw_audio_without_array_key(
        self,
        get_cached_dataset_mock,
        decode_audio_item_mock,
        calculate_similarity_mock,
        generate_dashboard_outputs_mock,
        save_results_csv_mock,
    ):
        set_global_results([
            {
                "id": 0,
                "path": "bad.wav",
                "ref_text": "ref",
                "score": 80,
                "verification_status": "incorrect",
                "model_results": {},
            }
        ])
        get_cached_dataset_mock.return_value = [
            {"audio": {"path": "bad.wav", "bytes": b"raw-audio"}}
        ]
        hf_client = Mock()
        hf_client.transcribe_batch.return_value = {0: "hyp"}

        outputs = _run_hf_recheck_analysis(
            hf_client=hf_client,
            model_name="SeamlessM4T-v2 (HF)",
            dataset_name="demo/dataset",
            limit_files=0,
            analysis_scope=ANALYSIS_SCOPE_PROBLEMATIC,
            similarity_threshold=95,
            hf_token=None,
            progress=lambda *args, **kwargs: None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        batch_audio = hf_client.transcribe_batch.call_args.args[0]
        self.assertEqual(batch_audio, [(0, [0.9], 16000)])
        decode_audio_item_mock.assert_called_once()
        calculate_similarity_mock.assert_called_once_with("ref", "hyp")
        generate_dashboard_outputs_mock.assert_called_once_with(95)
        save_results_csv_mock.assert_called_once()

    @patch("analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.utils.calculate_similarity", return_value=(97, "ref", "hyp"))
    @patch("analysis.standard.utils.decode_audio_item", return_value=([0.9], 16000, "pending.wav"))
    @patch("analysis.standard._load_analysis_dataset")
    def test_vertex_batch_pending_scope_decodes_raw_audio_bytes(
        self,
        load_analysis_dataset_mock,
        decode_audio_item_mock,
        calculate_similarity_mock,
        generate_dashboard_outputs_mock,
    ):
        set_global_results([
            {
                "id": 0,
                "path": "pending.wav",
                "ref_text": "pending",
                "score": 0,
                "verification_status": "pending",
                "model_results": {},
            }
        ])
        load_analysis_dataset_mock.return_value = [
            {"audio": {"path": "pending.wav", "bytes": b"raw-audio"}}
        ]
        gemini_tool = Mock()
        gemini_tool.transcribe_audio_batch.return_value = {0: "batch text"}

        outputs = _run_vertex_batch_analysis(
            gemini_tool=gemini_tool,
            model_name="gemini-2.5-flash-lite",
            dataset_name="demo/dataset",
            limit_files=0,
            analysis_scope=ANALYSIS_SCOPE_PENDING,
            similarity_threshold=95,
            gen_config={"temperature": 0.2},
            hf_token=None,
            progress=lambda *args, **kwargs: None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        batch_records = gemini_tool.transcribe_audio_batch.call_args.args[1]
        self.assertEqual(batch_records[0]["audio_array"], [0.9])
        self.assertEqual(batch_records[0]["sampling_rate"], 16000)
        decode_audio_item_mock.assert_called_once()
        calculate_similarity_mock.assert_called_once_with("pending", "batch text")
        generate_dashboard_outputs_mock.assert_called_once_with(95)

    @patch("analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.utils.calculate_similarity", return_value=(96, "ref", "hyp"))
    @patch("analysis.standard.utils.decode_audio_item", return_value=([0.4], 16000, "bad.wav"))
    @patch("analysis.standard._load_analysis_dataset")
    def test_vertex_batch_problematic_scope_rechecks_existing_results(
        self,
        load_analysis_dataset_mock,
        decode_audio_item_mock,
        calculate_similarity_mock,
        generate_dashboard_outputs_mock,
    ):
        set_global_results([
            {
                "id": 0,
                "path": "bad.wav",
                "ref_text": "ref",
                "score": 80,
                "verification_status": "incorrect",
                "model_results": {},
            },
            {
                "id": 1,
                "path": "ok.wav",
                "ref_text": "ok",
                "score": 99,
                "verification_status": "correct",
                "model_results": {},
            },
        ])
        load_analysis_dataset_mock.return_value = [
            {"audio": {"path": "bad.wav", "bytes": b"raw-audio"}},
            {"audio": {"path": "ok.wav", "bytes": b"raw-audio"}},
        ]
        gemini_tool = Mock()
        gemini_tool.transcribe_audio_batch.return_value = {0: "hyp"}

        outputs = _run_vertex_batch_analysis(
            gemini_tool=gemini_tool,
            model_name="gemini-2.5-flash-lite",
            dataset_name="demo/dataset",
            limit_files=0,
            analysis_scope=ANALYSIS_SCOPE_PROBLEMATIC,
            similarity_threshold=95,
            gen_config={"temperature": 0.2},
            hf_token=None,
            progress=lambda *args, **kwargs: None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        batch_records = gemini_tool.transcribe_audio_batch.call_args.args[1]
        self.assertEqual([record["id"] for record in batch_records], [0])
        self.assertEqual(batch_records[0]["audio_array"], [0.4])
        decode_audio_item_mock.assert_called_once()
        calculate_similarity_mock.assert_called_once_with("ref", "hyp")
        generate_dashboard_outputs_mock.assert_called_once_with(95)

    @patch("analysis.standard.generate_dashboard_outputs", return_value=("stats", "flagged", "table"))
    @patch("analysis.standard.utils.calculate_similarity", return_value=(98, "ref", "hyp"))
    @patch("analysis.standard.utils.decode_audio_item", return_value=([0.7], 22050, "fresh.wav"))
    @patch("analysis.standard._load_analysis_dataset")
    def test_vertex_batch_full_run_decodes_raw_audio_bytes(
        self,
        load_analysis_dataset_mock,
        decode_audio_item_mock,
        calculate_similarity_mock,
        generate_dashboard_outputs_mock,
    ):
        clear_global_results()
        load_analysis_dataset_mock.return_value = [
            {
                "audio": {"path": "fresh.wav", "bytes": b"raw-audio"},
                "sentence": "fresh ref",
            }
        ]
        gemini_tool = Mock()
        gemini_tool.transcribe_audio_batch.return_value = {0: "fresh text"}

        outputs = _run_vertex_batch_analysis(
            gemini_tool=gemini_tool,
            model_name="gemini-2.5-flash-lite",
            dataset_name="demo/dataset",
            limit_files=0,
            analysis_scope=ANALYSIS_SCOPE_ALL,
            similarity_threshold=95,
            gen_config={"temperature": 0.2},
            hf_token=None,
            progress=lambda *args, **kwargs: None,
        )

        self.assertEqual(outputs, ("stats", "flagged", "table"))
        batch_records = gemini_tool.transcribe_audio_batch.call_args.args[1]
        self.assertEqual(batch_records[0]["audio_array"], [0.7])
        self.assertEqual(batch_records[0]["sampling_rate"], 22050)
        decode_audio_item_mock.assert_called_once()
        calculate_similarity_mock.assert_called_once_with("fresh ref", "fresh text")
        generate_dashboard_outputs_mock.assert_called_once_with(95)


if __name__ == "__main__":
    unittest.main()
