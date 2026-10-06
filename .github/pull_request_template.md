## Summary

<!-- Briefly describe what this PR changes and why. -->

## Related issues

<!-- Link issues this addresses, e.g. Fixes #123 or Relates to #456. -->

## Type of change

- [ ] Bug fix
- [ ] New feature or enhancement
- [ ] Documentation
- [ ] Refactor or internal cleanup
- [ ] Other (describe below)

## Changes

<!-- What did you change? Call out anything reviewers should focus on. -->


## Testing

<!-- How did you verify this works? Delete sections that do not apply. -->

### Python (`Resources/python`, `Test`)

- [ ] Installed test dependencies: `pip install --group test -e "./Resources/python[all]"`
- [ ] Ran: `python -m pytest Test --import-mode=importlib -n 0`
- [ ] Not applicable

### Godot (`Godot`)

- [ ] Ran clang-format 17 on changed C++ files
- [ ] Built the relevant GDExtension target
- [ ] Ran relevant Catch2 and/or GdUnit4 tests
- [ ] Not applicable

## Checklist

- [ ] Code follows `Godot/docs/development_contract.md` and [Black](https://black.readthedocs.io/) for Python
- [ ] Comments / docstrings added or updated where behavior is non-obvious
- [ ] Applicable README, development-contract, or Sphinx documentation updated
- [ ] No unrelated changes included in this PR
- [ ] New source and build-script files include the standard AMD copyright notice
- [ ] Generated or vendored files were not edited manually
