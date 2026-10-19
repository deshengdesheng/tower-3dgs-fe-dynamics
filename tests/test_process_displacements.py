import tempfile
import subprocess
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from workflow.paper_stages.process_displacements import process_points  # noqa: E402

SCRIPT = REPO_ROOT / "workflow" / "paper_stages" / "process_displacements.py"


class ProcessDisplacementsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.reference = self.root / "reference.txt"
        self.deformed = self.root / "deformed.txt"
        self.output = self.root / "with_displacement.txt"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_inputs(self, reference_rows, deformed_rows):
        self.reference.write_text("# idx x y z\n" + "\n".join(reference_rows) + "\n", encoding="utf-8")
        self.deformed.write_text("# idx x y z\n" + "\n".join(deformed_rows) + "\n", encoding="utf-8")

    def test_matches_rows_by_node_id_when_file_order_differs(self):
        self.write_inputs(
            ["10 0 0 0", "20 1 0 0"],
            ["20 1 2 0", "10 0 1 0"],
        )

        self.assertTrue(process_points(self.reference, self.deformed, self.output))
        rows = [line.split() for line in self.output.read_text(encoding="utf-8").splitlines()[1:]]
        self.assertEqual(rows, [
            ["10", "0.000000", "0.000000", "0.000000", "0.000000", "1.000000", "0.000000"],
            ["20", "1.000000", "0.000000", "0.000000", "0.000000", "2.000000", "0.000000"],
        ])

    def test_fails_when_node_id_sets_differ_without_writing_output(self):
        self.write_inputs(["10 0 0 0", "20 1 0 0"], ["10 0 1 0", "30 1 2 0"])

        with self.assertRaisesRegex(ValueError, "节点 ID 集不一致"):
            process_points(self.reference, self.deformed, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_exits_nonzero_when_node_id_sets_differ(self):
        self.write_inputs(["10 0 0 0", "20 1 0 0"], ["10 0 1 0", "30 1 2 0"])

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--input1", str(self.reference),
             "--input2", str(self.deformed), "--output", str(self.output)],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("节点 ID 集不一致", result.stderr)
        self.assertFalse(self.output.exists())

    def test_fails_on_duplicate_node_ids(self):
        self.write_inputs(["10 0 0 0", "10 1 0 0"], ["10 0 1 0", "10 1 1 0"])

        with self.assertRaisesRegex(ValueError, "节点 ID 重复"):
            process_points(self.reference, self.deformed, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
