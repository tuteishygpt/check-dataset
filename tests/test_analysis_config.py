import unittest

from analysis.smart import build_smart_generation_config
from analysis.standard import build_model_generation_config, normalize_execution_mode


class AnalysisConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
