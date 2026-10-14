import json
from pathlib import Path
import tempfile
import unittest

from cambacica.tokenizer.experiments import _open_eval_index


class EvaluationIndexReuseTests(unittest.TestCase):
    def test_reuse_and_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            identity = {"schema_version": "test-v1", "seed": 123}

            db = _open_eval_index(path, identity)
            db.execute(
                "INSERT INTO meta(key,value) VALUES (?,?)",
                ("assignment_checksum_sha256", json.dumps({"sha256": "test"})),
            )
            db.commit()
            db.close()

            # Valid checkpoint must be reusable.
            reused = _open_eval_index(path, identity)
            reused.close()

            # Changed scientific identity must still be rejected.
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                _open_eval_index(
                    path,
                    {"schema_version": "test-v1", "seed": 456},
                )


if __name__ == "__main__":
    unittest.main()
