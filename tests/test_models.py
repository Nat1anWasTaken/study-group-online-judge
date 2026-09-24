import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from judge.models import (
    JudgeResult,
    Resources,
    SubJudge,
    SubJudgeBackend,
    Submission,
    TestResult,
)


class SubmissionTests(unittest.TestCase):
    def test_accepts_an_exact_commit_sha(self) -> None:
        submission = Submission(
            repo_url="https://github.com/cerulean-works/example.git",
            commit_sha="a" * 40,
            task_id="example",
            github_actor="student",
        )

        self.assertEqual(submission.commit_sha, "a" * 40)

    def test_rejects_a_branch_name_instead_of_a_commit_sha(self) -> None:
        with self.assertRaises(ValidationError):
            Submission(
                repo_url="https://github.com/cerulean-works/example.git",
                commit_sha="main",
                task_id="example",
                github_actor="student",
            )

    def test_rejects_a_non_github_repository(self) -> None:
        with self.assertRaises(ValidationError):
            Submission(
                repo_url="https://example.com/owner/repository.git",
                commit_sha="a" * 40,
                task_id="example",
                github_actor="student",
            )


class JudgeResultTests(unittest.TestCase):
    def test_supports_pass_fail_results(self) -> None:
        result = JudgeResult(
            passed=False,
            tests=[TestResult(name="shape", passed=False, message="wrong shape")],
        )

        self.assertFalse(result.passed)
        self.assertIsNone(result.score)

    def test_supports_scored_results(self) -> None:
        result = JudgeResult(
            score=18.42,
            metrics={"perplexity": 18.42},
        )

        self.assertIsNone(result.passed)
        self.assertEqual(result.score, 18.42)

    def test_result_collections_are_not_shared(self) -> None:
        first = JudgeResult()
        second = JudgeResult()

        first.metrics["accuracy"] = 1.0
        first.tests.append(TestResult(name="example", passed=True))

        self.assertEqual(second.metrics, {})
        self.assertEqual(second.tests, [])


class ResourcesTests(unittest.TestCase):
    def test_rejects_invalid_resource_limits(self) -> None:
        with self.assertRaises(ValidationError):
            Resources(cpus=0)

    def test_is_immutable(self) -> None:
        resources = Resources()
        field = "gpus"

        with self.assertRaises(ValidationError):
            setattr(resources, field, 1)


class SubJudgeTests(unittest.TestCase):
    def test_rejects_invalid_identity_or_gpu_capacity(self) -> None:
        valid = {
            "id": "nano4",
            "backend": SubJudgeBackend.SLURM,
            "task_ids": ["gpu-lab"],
            "max_gpus": 8,
            "judge_revision": "test-revision",
            "registered_at": datetime.now(UTC),
        }

        for invalid in ({"id": "../nano4"}, {"max_gpus": -1}, {"task_ids": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                SubJudge.model_validate(valid | invalid)


if __name__ == "__main__":
    unittest.main()
