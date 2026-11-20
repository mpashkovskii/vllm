# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Label PRs whose changes are contained in one model folder (RFC #59665).

A PR is model-scoped when every file it changes is under one
`vllm/models/<model>/<vendor>/` folder or under `tests/models/<model>/`.
It then cannot change another model, another vendor's code or shared code, so
the CODEOWNERS of that folder are the reviewers who can judge it. The script
labels such PRs, keeps one comment with the folder's owners and whether one of
them approved, and removes both when a later push leaves the folder.

CODEOWNERS is read from the default branch checked out by the workflow, never
from the PR, so a PR cannot name its own owners.
"""

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import regex as re
from run_ci_command import GitHubClient, resolve_workflow_run_pr

SCOPED_LABEL = "model-scoped"
APPROVED_LABEL = "model-owner-approved"
COMMENT_MARKER = "<!-- vllm-model-scope -->"
SHARED_MODEL_DIRS = frozenset({"common"})
MAX_LISTED_FILES = 5

MODEL_CODE_RE = re.compile(r"^vllm/models/(?P<model>[^/]+)/(?P<vendor>[^/]+)/")


@dataclass(frozen=True)
class Scope:
    model: str
    vendor: str

    @property
    def folder(self) -> str:
        return f"vllm/models/{self.model}/{self.vendor}/"

    @property
    def tests(self) -> str:
        return f"tests/models/{self.model}/"


def find_scope(paths: Sequence[str]) -> tuple[Scope | None, list[str]]:
    """Returns the PR's model folder, or None and the files outside it."""
    scopes = set()
    for path in paths:
        match = MODEL_CODE_RE.match(path)
        if match and not SHARED_MODEL_DIRS & {match["model"], match["vendor"]}:
            scopes.add(Scope(match["model"], match["vendor"]))
    if len(scopes) > 1:
        return None, sorted(scope.folder for scope in scopes)
    if not scopes:
        return None, list(paths)
    scope = scopes.pop()
    outside = [
        path for path in paths if not path.startswith((scope.folder, scope.tests))
    ]
    return (None if outside else scope), outside


def codeowners_pattern(pattern: str) -> re.Pattern[str]:
    """Translates a CODEOWNERS path pattern to a regex over repo paths."""
    anchored = "/" in pattern.rstrip("/")
    body = pattern.strip("/")
    parts = []
    i = 0
    while i < len(body):
        if body.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif body.startswith("**", i):
            parts.append(".*")
            i += 2
        elif body[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif body[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(body[i]))
            i += 1
    prefix = "" if anchored else "(?:.*/)?"
    # `docs/*` owns the files in docs/ but not in its subfolders.
    subtree = "" if pattern.endswith("/*") else "(?:/.*)?"
    return re.compile(f"^{prefix}{''.join(parts)}{subtree}$")


@dataclass(frozen=True)
class CodeOwners:
    rules: tuple[tuple[re.Pattern[str], frozenset[str]], ...]

    @classmethod
    def parse(cls, text: str) -> "CodeOwners":
        rules = []
        for line in text.splitlines():
            fields = line.split("#", 1)[0].split()
            if fields:
                owners = frozenset(
                    owner.removeprefix("@").lower() for owner in fields[1:]
                )
                rules.append((codeowners_pattern(fields[0]), owners))
        return cls(tuple(rules))

    def owners(self, path: str) -> frozenset[str]:
        for pattern, owners in reversed(self.rules):
            if pattern.match(path):
                return owners
        return frozenset()

    def common_owners(self, paths: Iterable[str]) -> frozenset[str]:
        owner_sets = [self.owners(path) for path in paths]
        return frozenset.intersection(*owner_sets) if owner_sets else frozenset()


def latest_review_states(reviews: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each reviewer's latest approval, change request or dismissal."""
    states: dict[str, str] = {}
    for review in reviews:
        state = review.get("state")
        user = review.get("user") or {}
        if state in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"} and user:
            states[str(user["login"]).lower()] = state
    return states


@dataclass(frozen=True)
class Verdict:
    scope: Scope | None
    outside: tuple[str, ...]
    owners: frozenset[str]
    review_states: Mapping[str, str]

    @cached_property
    def approved_by(self) -> list[str]:
        return sorted(
            user
            for user, state in self.review_states.items()
            if state == "APPROVED" and user in self.owners
        )

    @cached_property
    def changes_requested_by(self) -> list[str]:
        return sorted(
            user
            for user, state in self.review_states.items()
            if state == "CHANGES_REQUESTED"
        )

    @property
    def owner_approved(self) -> bool:
        return bool(self.scope and self.approved_by and not self.changes_requested_by)


def evaluate(
    paths: Sequence[str],
    codeowners: CodeOwners,
    reviews: Sequence[Mapping[str, Any]],
) -> Verdict:
    scope, outside = find_scope(paths)
    owners = (
        codeowners.common_owners(p for p in paths if p.startswith(scope.folder))
        if scope
        else frozenset()
    )
    return Verdict(scope, tuple(outside), owners, latest_review_states(reviews))


def mentions(users: Iterable[str]) -> str:
    return ", ".join(f"@{user}" for user in users)


def render_comment(verdict: Verdict) -> str:
    if verdict.scope is None:
        listed = "\n".join(f"- `{path}`" for path in verdict.outside[:MAX_LISTED_FILES])
        more = len(verdict.outside) - MAX_LISTED_FILES
        if more > 0:
            listed += f"\n- and {more} more"
        return (
            f"{COMMENT_MARKER}\nThis PR is no longer model-scoped: it changes "
            f"paths outside a single model folder.\n\n{listed}\n"
        )
    scope = verdict.scope
    lines = [
        COMMENT_MARKER,
        (
            f"**Model-scoped:** this PR only changes `{scope.folder}` and "
            f"`{scope.tests}`, so it cannot affect other models, other vendors "
            "or shared code (RFC #59665)."
        ),
        "",
    ]
    if not verdict.owners:
        lines.append(
            f"- No CODEOWNERS entry covers `{scope.folder}`; "
            "a maintainer review is required."
        )
    else:
        lines.append(f"- Folder owners: {mentions(sorted(verdict.owners))}")
        approval = (
            f"approved by {mentions(verdict.approved_by)}"
            if verdict.approved_by
            else "waiting for a folder owner"
        )
        lines.append(f"- Owner approval: {approval}")
    if verdict.changes_requested_by:
        lines.append(f"- Changes requested by {mentions(verdict.changes_requested_by)}")
    return "\n".join(lines) + "\n"


class ModelScopeGitHubClient(GitHubClient):
    def add_labels(self, number: int, labels: Sequence[str]) -> None:
        self._request(
            self._repo_path(f"/issues/{number}/labels"),
            body={"labels": list(labels)},
            method="POST",
        )

    def remove_label(self, number: int, label: str) -> None:
        self._request(
            self._repo_path(f"/issues/{number}/labels/{label}"),
            method="DELETE",
        )

    def update_comment(self, comment_id: int, body: str) -> None:
        self._request(
            self._repo_path(f"/issues/comments/{comment_id}"),
            body={"body": body},
            method="PATCH",
        )


def apply(pr: Mapping[str, Any], verdict: Verdict, github: Any) -> None:
    number = pr["number"]
    labels = {label["name"] for label in pr.get("labels", [])}
    wanted = set()
    if verdict.scope:
        wanted.add(SCOPED_LABEL)
    if verdict.owner_approved:
        wanted.add(APPROVED_LABEL)
    if wanted - labels:
        github.add_labels(number, sorted(wanted - labels))
    for label in sorted({SCOPED_LABEL, APPROVED_LABEL} & labels - wanted):
        github.remove_label(number, label)

    previous = next(
        (
            comment
            for comment in github.list_issue_comments(number)
            if COMMENT_MARKER in comment.get("body", "")
        ),
        None,
    )
    if verdict.scope is None and previous is None:
        return
    body = render_comment(verdict)
    if previous is None:
        github.add_comment(number, body)
    elif previous["body"] != body:
        github.update_comment(previous["id"], body)


def run(pr: Mapping[str, Any], github: Any, codeowners: CodeOwners) -> Verdict:
    number = pr["number"]
    verdict = evaluate(
        github.list_pr_files(number), codeowners, github.list_reviews(number)
    )
    apply(pr, verdict, github)
    return verdict


def main() -> None:
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
        event = json.load(event_file)
    github = ModelScopeGitHubClient(
        os.environ.get("GH_TOKEN", ""), os.environ["GITHUB_REPOSITORY"]
    )
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_run":
        pr = resolve_workflow_run_pr(event["workflow_run"], github)
        if pr is None:
            print("Could not resolve an open PR for the approval workflow run.")
            return
    else:
        pr = event["pull_request"]
    with open(
        os.environ.get("CODEOWNERS_PATH", ".github/CODEOWNERS"), encoding="utf-8"
    ) as codeowners_file:
        codeowners = CodeOwners.parse(codeowners_file.read())
    verdict = run(pr, github, codeowners)
    print(
        f"PR #{pr['number']}: scope={verdict.scope}, "
        f"owners={sorted(verdict.owners)}, approved_by={verdict.approved_by}"
    )


if __name__ == "__main__":
    main()
