---
name: release
description: Promote the engine from dev to main and, when Leo asks for a release (or a downstream pack's session asks for a patch), tag it and tell downstream packs the tag. Covers the version bump, make release-check, the pull-request squash promotion, and the tag and push. Use for "release", "promote to main" or "tag vX.Y.Z".
---

# Release the engine

Keep this file current: change it in the same commit as any change to the
steps it names (`Makefile` targets, `hooks/pre-commit`, `hooks/pre-merge-commit`, the
version source).
`tests/test_release_skill.py` checks the paths and targets it names.

Promote only when Leo asks. Tag and push only when Leo asks for a release.
One exception: when a downstream pack's session asks for an engine change, make
it and release it as a patch (vX.Y.Z+1) without asking Leo, provided every gate
below passes. A minor release, a GitHub setting, or a change in another repo
still needs Leo's go.

Never copy a file from a live pack or a data directory into this repo, at a
release or otherwise: the history example's tables are written by hand, and
`tests/test_no_live_data.py` fails on anything that looks copied.

## 1. Prepare on `dev`

1. Confirm `git status --short` is clean and the branch is `dev`.
2. For a release, bump `__version__` in `src/universe_builder/__init__.py`
   (patch: fixes; minor: a feature, a removal or a changed default). It is the
   only version source. Run `make install` after the bump, or
   `test_the_installed_metadata_agrees` fails on the stale metadata. Commit
   the bump on `dev` as "vX.Y.Z".
3. Update the `docs/architecture/` parts the release touches.

## 2. Gate

Run `make release-check` to a file; act on its exit code; read the tail only
on failure. It runs the suite and flake8. Do not promote on a non-zero exit.

Then have a second agent read `git diff <last tag>..dev`, the commit messages,
and the pull request title and body you will use in step 3, for live data the
guards cannot see: corpus counts, costs, run dates, subject lists or table
lines from a pack, retyped or summarised rather than copied. No hook checks
the pull request text. Take out anything it finds before promoting; this repo
is public.

## 3. Promote through a pull request

`main` takes changes only through a squash-merged pull request from `dev`:
its branch protection requires one, with the checks `lint`, `test (3.10)`,
`test (3.12)` and `test (3.14)` green and the branch up to date, for admins
too. A change to the CI matrix renames those checks: update the required
checks in `main`'s protection in the same change, or every merge blocks.

```bash
git push origin dev
gh pr create --base main --head dev --title "vX.Y.Z: <summary>" --body "<what changed for packs>"
gh pr checks <N> --watch
gh pr merge <N> --squash --match-head-commit "$(git rev-parse dev)" --subject "vX.Y.Z: <summary>" --body ""
git fetch origin
git merge --no-edit origin/main      # on dev
git diff --quiet origin/main dev     # must exit 0: the two trees match
git branch -f main origin/main
git push origin dev
```

`--match-head-commit` refuses the merge if another agent pushed to `dev`
during the review. The squash body is left blank, or GitHub would paste every
`dev` commit since the first release into it.

Merge `main` back into `dev`; never `git reset --hard main`. The merge
changes no files, keeps any `dev` commit or uncommitted edit another agent
made meanwhile, and lets the next pull request diff only the work since this
promotion. `git log --first-parent dev` hides the doubled history.

Local `main` only follows GitHub's: `hooks/pre-commit` and
`hooks/pre-merge-commit` (installed by `make hooks`) refuse any commit on it.
Run `make test` on `dev` after the merge.

## 4. Tag (a release only)

Tag the commit GitHub made, not a local `main` that may be stale:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z" "$(gh pr view <N> --json mergeCommit -q .mergeCommit.oid)"
git push origin vX.Y.Z
```

Never force-push either branch, and never move or delete a pushed tag (a
ruleset refuses both): a pack installs whatever `@vX.Y.Z` points at. A
refused `dev` push means someone else pushed: `git fetch origin`,
`git merge origin/dev`, push again.

## 5. Tell downstream packs

A pack pins the engine by tag. After the push returns, send each pack's
session the tag and what changed for packs; the pack bumps its own pin from
its own repo.
