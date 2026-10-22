---
name: release
description: Promote the engine from dev to main and, when Leo asks for a release, tag it and tell downstream packs the tag. Covers the version bump, make release-check, the squash promotion, and the tag and push. Use for "release", "promote to main" or "tag vX.Y.Z".
---

# Release the engine

Keep this file current: change it in the same commit as any change to the
steps it names (`Makefile` targets, `hooks/pre-commit`, the version source).
`tests/test_release_skill.py` checks the paths and targets it names.

Promote only when Leo asks. Tag and push only when Leo asks for a release.

Never copy a file from a live pack or a data directory into this repo, at a
release or otherwise: the history example's tables are written by hand, and
`tests/test_no_live_data.py` fails on anything that looks copied.

## 1. Prepare on `dev`

1. Confirm `git status --short` is clean and the branch is `dev`.
2. For a release, bump `__version__` in `src/universe_builder/__init__.py`
   (patch: fixes; minor: a feature, a removal or a changed default). It is the
   only version source. Run `make install` after the bump, or
   `test_the_installed_metadata_agrees` fails on the stale metadata.
3. Update the `docs/architecture/` parts the release touches.

## 2. Gate

Run `make release-check` to a file; act on its exit code; read the tail only
on failure. It runs the suite and flake8. Do not promote on a non-zero exit.

## 3. Promote

```bash
git checkout main
git merge --squash dev
git commit            # "vX.Y.Z: <summary>" for a release; the hook runs flake8
git checkout dev
git reset --hard main
```

`hooks/pre-commit` (installed by `make hooks`) refuses a commit on `main`
that is not a squash. Run `make test` on `dev` after the reset.

## 4. Tag and push (a release only)

```bash
git tag vX.Y.Z main
git push origin main vX.Y.Z
git push origin dev
```

If `git push origin dev` is refused because `dev` was pushed before the
reset, ask Leo before `--force-with-lease`.

## 5. Tell downstream packs

A pack pins the engine by tag. After the push returns, send each pack's
session the tag and what changed for packs; the pack bumps its own pin from
its own repo.
