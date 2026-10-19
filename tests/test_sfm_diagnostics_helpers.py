import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "revision_experiments"))

from sfm_quality_group_diagnostics import (  # noqa: E402
    has_camera_prefix,
    parse_cc_export_image_keys,
    source_key,
)


class SfMDiagnosticsHelperTest(unittest.TestCase):
    def test_camera_prefix_is_explicit_and_case_insensitive(self):
        self.assertTrue(has_camera_prefix("CAM01_frame.jpg", "cam01"))
        self.assertFalse(has_camera_prefix("CAM02_frame.jpg", "cam01"))
        self.assertFalse(has_camera_prefix("CAM01_frame.jpg", None))
        self.assertEqual(source_key("CAM01/frame.jpg", "cam01"), "CAM01_frame.jpg")
        self.assertEqual(source_key("CAM01_frame.jpg", "CAM01"), "CAM01_frame.jpg")

    def test_xml_paths_use_optional_dataset_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "export.xml"
            xml_path.write_text(
                '<image path="C:/capture/session/CAM01_frame.jpg"/>',
                encoding="utf-8",
            )
            keys, counts = parse_cc_export_image_keys(
                xml_path,
                dataset_marker="session",
                camera_prefix="CAM01",
            )

        self.assertEqual(keys, {"CAM01_frame.jpg"})
        self.assertEqual(counts["xml_camera_prefix_paths"], 1)
        self.assertEqual(counts["xml_other_paths"], 0)


if __name__ == "__main__":
    unittest.main()
