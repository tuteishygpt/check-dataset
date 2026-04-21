import unittest

from analysis.smart import build_smart_generation_config
from analysis.standard import build_model_generation_config


class AnalysisConfigTests(unittest.TestCase):
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
