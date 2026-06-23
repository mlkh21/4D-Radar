import os
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MiniScriptsProtocolTest(unittest.TestCase):
    def _read(self, relative_path):
        with open(os.path.join(ROOT, relative_path), "r", encoding="utf-8") as f:
            return f.read()

    def test_training_script_supports_sensor_aware_root_and_ir_linking(self):
        script = self._read("test/mini-test/train_minimal.sh")

        self.assertIn('PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-', script)
        self.assertIn('SRC_IR_DIR="${SRC_SCENE_DIR}/ir_image"', script)
        self.assertIn('DST_IR_DIR="${DST_SCENE_DIR}/ir_image"', script)
        self.assertIn('ln -s "${SRC_IR_PATH}"', script)
        self.assertIn('preprocess_policy.json', script)

    def test_inference_script_accepts_matching_data_and_result_roots(self):
        script = self._read("test/mini-test/inference_minimal.sh")

        self.assertIn('PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-', script)
        self.assertIn('MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-', script)
        self.assertIn('--save_uncertainty', script)


if __name__ == "__main__":
    unittest.main()
