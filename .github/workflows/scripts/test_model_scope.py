# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from typing import Any

from model_scope import (
    APPROVED_LABEL,
    COMMENT_MARKER,
    SCOPED_LABEL,
    CodeOwners,
    Scope,
    evaluate,
    find_scope,
    run,
)

# An excerpt of .github/CODEOWNERS.
REPO_CODEOWNERS = CodeOwners.parse("""
/vllm/models/deepseek_v4 @zyongye
/vllm/models/kimi_k3/nvidia/kda.py @gau-nernst
/vllm/**/*rocm* @tjtanaa @dllehr-amd @shen-shanshan
/vllm/models/*/amd @tjtanaa @dllehr-amd @hongxiayang
/vllm/models/deepseek_v4/amd @tjtanaa @dllehr-amd @hongxiayang @zyongye
""")
AMD_OWNERS = {"tjtanaa", "dllehr-amd", "hongxiayang"}
KIMI_AMD = [
    "vllm/models/kimi_k3/amd/specialized_decode/runtime/decode.py",
    "tests/models/kimi_k3/amd/test_numerics.py",
]


def review(user: str, state: str) -> dict[str, Any]:
    return {"user": {"login": user}, "state": state}


class FakeGitHub:
    def __init__(self, files: list[str], reviews: list[dict[str, Any]]) -> None:
        self.files = files
        self.reviews = reviews
        self.labels: set[str] = set()
        self.comments: dict[int, str] = {}

    def pr(self) -> dict[str, Any]:
        return {"number": 7, "labels": [{"name": name} for name in self.labels]}

    def list_pr_files(self, number: int) -> list[str]:
        return self.files

    def list_reviews(self, number: int) -> list[dict[str, Any]]:
        return self.reviews

    def list_issue_comments(self, number: int) -> list[dict[str, Any]]:
        return [{"id": i, "body": body} for i, body in self.comments.items()]

    def add_labels(self, number: int, labels: list[str]) -> None:
        self.labels.update(labels)

    def remove_label(self, number: int, label: str) -> None:
        self.labels.remove(label)

    def add_comment(self, number: int, body: str) -> None:
        self.comments[len(self.comments)] = body

    def update_comment(self, comment_id: int, body: str) -> None:
        self.comments[comment_id] = body


class FindScopeTest(unittest.TestCase):
    def test_model_folder_and_its_tests_are_one_scope(self) -> None:
        self.assertEqual(find_scope(KIMI_AMD), (Scope("kimi_k3", "amd"), []))

    def test_files_outside_the_folder_break_the_scope(self) -> None:
        for extra in (
            ".buildkite/test_areas/models_basic.yaml",
            "vllm/_aiter_ops.py",
            "vllm/models/kimi_k3/nvidia/model.py",
            "vllm/models/kimi_k3/common/layers.py",
            "tests/models/deepseek_v4/amd/test_mono.py",
        ):
            with self.subTest(extra=extra):
                self.assertEqual(find_scope([*KIMI_AMD, extra])[0], None)

    def test_shared_model_code_is_never_a_scope(self) -> None:
        scope, _ = find_scope(["vllm/models/common/ops/fused_qk_rmsnorm.py"])
        self.assertIsNone(scope)

    def test_tests_alone_are_not_a_scope(self) -> None:
        self.assertIsNone(find_scope(KIMI_AMD[1:])[0])


class CodeOwnersTest(unittest.TestCase):
    def test_repo_codeowners_last_matching_rule_wins(self) -> None:
        for path, owners in (
            ("vllm/models/kimi_k3/amd/model.py", AMD_OWNERS),
            ("vllm/models/deepseek_v4/amd/model.py", AMD_OWNERS | {"zyongye"}),
            ("vllm/models/deepseek_v4/nvidia/model.py", {"zyongye"}),
            ("vllm/models/kimi_k3/nvidia/kda.py", {"gau-nernst"}),
            ("vllm/models/muse_spark/nvidia/model.py", set()),
        ):
            with self.subTest(path=path):
                self.assertEqual(REPO_CODEOWNERS.owners(path), owners)

    def test_pattern_semantics(self) -> None:
        owners = CodeOwners.parse(
            "docs/* @flat\n*.md @md\n/vllm/**/*rocm* @rocm\n/csrc @csrc\n"
        )
        self.assertEqual(owners.owners("docs/a.txt"), {"flat"})
        self.assertEqual(owners.owners("docs/sub/a.txt"), frozenset())
        self.assertEqual(owners.owners("docs/sub/a.md"), {"md"})
        self.assertEqual(owners.owners("vllm/platforms/rocm.py"), {"rocm"})
        self.assertEqual(owners.owners("csrc/rocm/x.cu"), {"csrc"})
        self.assertEqual(owners.owners("vendor/csrc/x.cu"), frozenset())


class ApprovalTest(unittest.TestCase):
    def verdict(self, *reviews: dict[str, Any]):
        return evaluate(KIMI_AMD, REPO_CODEOWNERS, list(reviews))

    def test_only_folder_owners_approve(self) -> None:
        self.assertFalse(self.verdict(review("someone", "APPROVED")).owner_approved)
        self.assertTrue(self.verdict(review("TJTANAA", "APPROVED")).owner_approved)

    def test_latest_review_state_counts(self) -> None:
        verdict = self.verdict(
            review("tjtanaa", "APPROVED"),
            review("tjtanaa", "COMMENTED"),
            review("tjtanaa", "DISMISSED"),
        )
        self.assertFalse(verdict.owner_approved)

    def test_open_change_request_blocks(self) -> None:
        verdict = self.verdict(
            review("tjtanaa", "APPROVED"), review("someone", "CHANGES_REQUESTED")
        )
        self.assertFalse(verdict.owner_approved)


class RunTest(unittest.TestCase):
    def test_labels_and_comment_follow_the_pr(self) -> None:
        github = FakeGitHub(KIMI_AMD, [review("dllehr-amd", "APPROVED")])
        run(github.pr(), github, REPO_CODEOWNERS)
        self.assertEqual(github.labels, {SCOPED_LABEL, APPROVED_LABEL})
        [comment] = github.comments.values()
        self.assertIn("approved by @dllehr-amd", comment)

        github.files = [*KIMI_AMD, "vllm/_aiter_ops.py"]
        run(github.pr(), github, REPO_CODEOWNERS)
        self.assertEqual(github.labels, set())
        [comment] = github.comments.values()
        self.assertIn("no longer model-scoped", comment)
        self.assertIn("`vllm/_aiter_ops.py`", comment)

    def test_unscoped_pr_gets_no_comment(self) -> None:
        github = FakeGitHub(["vllm/_aiter_ops.py"], [])
        run(github.pr(), github, REPO_CODEOWNERS)
        self.assertEqual((github.labels, github.comments), (set(), {}))

    def test_unchanged_verdict_does_not_rewrite_the_comment(self) -> None:
        github = FakeGitHub(KIMI_AMD, [])
        run(github.pr(), github, REPO_CODEOWNERS)
        github.update_comment = None  # type: ignore[assignment]
        run(github.pr(), github, REPO_CODEOWNERS)
        [comment] = github.comments.values()
        self.assertTrue(comment.startswith(COMMENT_MARKER))
        self.assertIn("waiting for a folder owner", comment)


if __name__ == "__main__":
    unittest.main()
