# AGENTS.md — working agreement for this repository

## Development workflow (required)

Work in small, traceable steps and follow this loop for every change:

1. **Implement deliberately** — make targeted changes in small, reviewable steps.
2. **Run the relevant tests** and check that existing functionality is preserved.
   Full suite: `MPLBACKEND=Agg PYTHONPATH=. python3 -m pytest -p no:warnings`
   (single files: `... python3 -m pytest tests/test_x.py -p no:warnings`).
3. **If tests pass**: give a short summary of the changes and name the changed files.
4. **Then** create a meaningful Git commit and push to the GitHub repository.
5. **If tests fail, unexpected changes appear, or correctness is uncertain**: do
   NOT push automatically. Report the problem first.
6. **Never** change credentials, Git remote settings, or repository permissions
   without explicit approval.

Only commit the files that belong to the current step — never `git add -A` blindly.

## Project scope (do not violate)

- This project is a **virtual learning simulation only**. Do not build or imply
  real rocket / engine / flight-controller capability.
- **Never invent test results.** Only report what actually ran.
- Physics is labelled as modelled where applicable; keep it traceable and honest.
