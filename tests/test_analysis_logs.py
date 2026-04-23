import unittest
import logging
import sys

from core import state


class AnalysisLogCaptureTests(unittest.TestCase):
    def setUp(self):
        state.clear_analysis_logs()

    def tearDown(self):
        state.clear_analysis_logs()

    def test_capture_analysis_prints_appends_lines_to_log_buffer(self):
        with state.capture_analysis_prints():
            print("alpha")
            print("beta", 42)

        self.assertEqual(
            state.get_analysis_logs_text(),
            "alpha\nbeta 42",
        )

    def test_install_global_log_capture_captures_stdout_stderr_and_logging(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_flag = state.log_capture_installed
        original_handler = state.analysis_log_handler

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level

        try:
            state.log_capture_installed = False
            state.analysis_log_handler = None
            state.install_global_log_capture()

            print("stdout line")
            sys.stderr.write("stderr line\n")
            logging.getLogger("test.logs").warning("logger line")

            logs = state.get_analysis_logs_text()
            self.assertIn("stdout line", logs)
            self.assertIn("stderr line", logs)
            self.assertIn("logger line", logs)
        finally:
            if state.analysis_log_handler is not None:
                root_logger.removeHandler(state.analysis_log_handler)
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            state.original_stdout = original_stdout
            state.original_stderr = original_stderr
            state.analysis_log_handler = original_handler
            state.log_capture_installed = original_flag


if __name__ == "__main__":
    unittest.main()
