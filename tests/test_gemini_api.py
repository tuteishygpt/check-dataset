import unittest
from unittest import mock

from gemini_api import (
    DEFAULT_TRANSCRIPTION_PROMPT,
    FLEX_REQUEST_HEADERS,
    GeminiIntegrator,
    build_generation_config,
    supports_flex_inference,
    validate_flex_inference,
    validate_vertex_environment,
)


class BuildGenerationConfigTests(unittest.TestCase):
    def test_build_generation_config_enables_flex_timeout(self):
        config = build_generation_config(temperature=0.3, thinking_budget=0, flex_mode=True)

        self.assertEqual(config["temperature"], 0.3)
        self.assertEqual(config["http_options"]["headers"], FLEX_REQUEST_HEADERS)
        self.assertEqual(config["http_options"]["api_version"], "v1")
        self.assertGreaterEqual(config["http_options"]["timeout"], 600000)
        self.assertNotIn("thinking_config", config)

    def test_build_generation_config_adds_thinking_when_requested(self):
        config = build_generation_config(temperature=0.1, thinking_budget=2048, flex_mode=False)

        self.assertEqual(
            config["thinking_config"],
            {"include_thoughts": True, "budget_tokens": 2048},
        )
        self.assertNotIn("service_tier", config)


class VertexEnvironmentTests(unittest.TestCase):
    def test_validate_vertex_environment_accepts_explicit_values(self):
        project, location = validate_vertex_environment(project="demo-project", location="global")
        self.assertEqual(project, "demo-project")
        self.assertEqual(location, "global")

    def test_supports_flex_inference_for_preview_models_on_global(self):
        self.assertTrue(supports_flex_inference("gemini-3.1-flash-lite-preview", location="global"))
        self.assertFalse(supports_flex_inference("gemini-2.5-flash-lite", location="global"))
        self.assertFalse(supports_flex_inference("gemini-3.1-flash-lite-preview", location="us-central1"))

    def test_validate_flex_inference_rejects_wrong_location(self):
        with self.assertRaises(ValueError):
            validate_flex_inference("gemini-3.1-flash-lite-preview", location="us-central1")

    def test_validate_flex_inference_rejects_unsupported_model(self):
        with self.assertRaises(ValueError):
            validate_flex_inference("gemini-2.5-flash-lite", location="global")


class GeminiIntegratorTests(unittest.TestCase):
    @mock.patch("gemini_api.genai.Client")
    def test_integrator_uses_vertex_client(self, client_cls):
        GeminiIntegrator(project="demo-project", location="global")

        client_cls.assert_called_once_with(
            vertexai=True,
            project="demo-project",
            location="global",
        )

    @mock.patch("gemini_api.types.Part.from_bytes")
    @mock.patch("gemini_api.genai.Client")
    def test_transcribe_audio_uses_inline_audio_prompt(self, client_cls, from_bytes):
        client = client_cls.return_value
        client.models.generate_content.return_value = mock.Mock(text=" transcript ")
        from_bytes.return_value = "audio-part"
        integrator = GeminiIntegrator(project="demo-project", location="global")

        result = integrator.transcribe_audio(
            "gemini-2.5-flash-lite",
            audio_array=[0.0, 0.1],
            sampling_rate=16000,
            config=build_generation_config(temperature=0.2, thinking_budget=0, flex_mode=True),
            prompt=DEFAULT_TRANSCRIPTION_PROMPT,
        )

        self.assertEqual(result, "transcript")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash-lite")
        self.assertEqual(kwargs["contents"][0], DEFAULT_TRANSCRIPTION_PROMPT)
        self.assertEqual(kwargs["contents"][1], "audio-part")
        self.assertEqual(kwargs["config"]["http_options"]["headers"], FLEX_REQUEST_HEADERS)


if __name__ == "__main__":
    unittest.main()
