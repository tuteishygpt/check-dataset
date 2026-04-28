import threading
import unittest

from core.state import (
    clear_global_results,
    get_analysis_running,
    get_global_results,
    get_stop_requested,
    set_analysis_running,
    set_global_results,
    set_stop_requested,
)


class StateThreadSafetyTests(unittest.TestCase):
    def tearDown(self):
        clear_global_results()
        set_stop_requested(False)
        set_analysis_running(False)

    def test_concurrent_set_global_results_no_corruption(self):
        errors = []

        def writer(value):
            try:
                set_global_results(value)
                result = get_global_results()
                assert isinstance(result, list)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=([{"id": 1}],))
        t2 = threading.Thread(target=writer, args=([{"id": 2}],))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertFalse(errors)

    def test_concurrent_stop_requested_no_corruption(self):
        errors = []

        def toggler(value):
            try:
                set_stop_requested(value)
                result = get_stop_requested()
                assert isinstance(result, bool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggler, args=(i % 2 == 0,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)

    def test_concurrent_analysis_running_no_corruption(self):
        errors = []

        def toggler(value):
            try:
                set_analysis_running(value)
                result = get_analysis_running()
                assert isinstance(result, bool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggler, args=(i % 2 == 0,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(errors)


if __name__ == "__main__":
    unittest.main()
