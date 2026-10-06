# Contributing to Schola for Godot

Thank you for your interest in improving Schola. This guide covers how to set up a development environment, follow project conventions, and submit changes.

For the Godot toolchain, build commands, and test commands, see the [Godot development guide](Godot/README.md). For binding engineering rules, see the [Godot development contract](Godot/docs/development_contract.md).

This file is the canonical source for repository-wide contribution workflow and hygiene. `Godot/README.md` owns executable development commands, while `Godot/docs/development_contract.md` owns binding implementation rules. The root README and files under `deliverables/` also contain course-required material and are not substitutes for contributor documentation.

## Ways to contribute

- **Bug reports** — Use the [bug report](.github/ISSUE_TEMPLATE/bug_report.yml) template and include reproduction steps and system information when you can.
- **Features and improvements** — Use the [feature or improvement](.github/ISSUE_TEMPLATE/feature_request.yml) template. You do not need to frame every idea as fixing a specific problem.
- **Pull requests** — Code, tests, and documentation fixes are welcome via PR. Fill out the [pull request template](.github/pull_request_template.md) when you open a PR.
- **Documentation** — User-facing changes should update the applicable README, Godot guide, development contract, and/or Sphinx guide under `Docs/Sphinx/`.

Discuss larger changes in an issue first if you are unsure about direction or scope; that helps avoid rework. For questions that are not a good fit for a public GitHub issue, contact the Schola team at [schola@amd.com](mailto:schola@amd.com).

## Development setup

1. Clone the repository and initialize its pinned submodules:

   ```bash
   git submodule update --init --recursive
   ```

2. Install the supported Godot, C++, Python, and SCons versions listed in [`Godot/README.md`](Godot/README.md#supported-toolchain).
3. Create a Python virtual environment and install the Godot build requirements:

   ```bash
   python -m pip install --require-hashes -r Godot/requirements.txt
   ```

4. Build the native extensions and open the project using the platform-specific commands in [`Godot/README.md`](Godot/README.md#build-and-run).

Contributors working on Python compatibility may also install the relevant package extras (`sb3`, `rllib`, `minari`, `docs`) described in `Resources/python/pyproject.toml`.

## Pull request workflow

1. Fork the repository and create a branch from `main`.
2. Make focused changes; keep unrelated edits out of the same PR.
3. Run the tests that apply to your changes (see [Testing](#testing) below).
4. Open a pull request against `main` and complete the PR template checklist.
5. Link related issues (for example `Fixes #123` or `Relates to #456`) so they close automatically when appropriate.

### Commit messages

We recommend [Conventional Commits](https://www.conventionalcommits.org/) so history stays scannable and release notes are easier to assemble. Use the form:

```text
<type>(<optional scope>): <short description>
```

We follow the angular convention for additional common types such as:

| Type | Use for |
| ---- | ------- |
| `feat` | New behavior or capability |
| `fix` | Bug fixes |
| `docs` | README, Sphinx, or other documentation |
| `test` | Tests only |
| `refactor` | Code changes that are not fixes or features |
| `build` | Build scripts, dependencies, or third-party rebuilds |
| `ci` | Continuous-integration workflows and checks |
| `chore` | Maintenance (tooling, configs) that does not fit above |

Optional scopes help when a change is localized, for example `godot`, `python`, `proto`, or `docs`.

Examples:

```text
fix(python): handle empty observation buffers in rollout worker
feat(godot): add discrete observation space
docs(godot): document the native test workflow
test(godot): cover environment reset behavior
```

Use the imperative mood in the subject line (“add feature”, not “added feature”). Add a body after a blank line when context, trade-offs, or breaking changes need explanation. Breaking changes can be called out with a `BREAKING CHANGE:` footer as described in the Conventional Commits spec.

## Coding standards

### C++

- Follow the conventions and module boundaries in the [Godot development contract](Godot/docs/development_contract.md#c-conventions).
- Format team-owned C++ with clang-format 17 and `Godot/.clang-format` as described in [`Godot/README.md`](Godot/README.md#format-c-code).
- Use Doxygen-style comments (`/** ... */`) for non-obvious APIs. In Visual Studio you can switch C++ comment style to Doxygen under **Tools → Options → Text Editor → C/C++ → Code Style → General**.

Place engine-independent C++ unit tests under `Godot/tests/unit/`. Place Godot scene, binding, and extension integration tests under `Godot/tests/integration/`.

### Python

- Format with [Black](https://black.readthedocs.io/). Run Black locally before pushing.
- Follow PEP 8 aside from Black’s formatting choices.
- Use [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for public APIs. Sphinx-compatible RST in docstrings is supported.

### Copyright headers

New source and build-script files should include the standard AMD copyright notice at the top, matching neighboring files. This applies to the Godot implementation and Python compatibility code:

- C++: `// Copyright (c) <year> Advanced Micro Devices, Inc. All Rights Reserved.`
- Python, SCons, and GDScript: `# Copyright (c) <year> Advanced Micro Devices, Inc. All Rights Reserved.`


### Protocol buffers and generated code

If you change `.proto` definitions, regenerate the existing upstream bindings from the plugin root:

```bash
schola compile-proto
```

The current command does not generate Godot bindings. Godot generation remains deferred until the US3 compatibility work selects compatible gRPC, Protocol Buffers, and generator versions. Do not hand-edit generated Python modules or generated Godot C++ bindings. The repository-level `Proto/` directory is the authoritative wire contract; do not create a Godot-specific copy. Godot generation rules and paths are defined in the [development contract](Godot/docs/development_contract.md#protocol-buffer-generation).

## Testing

The Godot implementation uses Catch2 for engine-independent C++ unit tests and GdUnit4 for Godot integration tests. The existing pytest suite covers Python behavior and compatibility.

### Godot C++ unit tests

Build and run the pinned Catch2 test target from the repository root:

```bash
scons -C Godot unit_tests platform=linux target=template_debug -j"$(nproc)"
./Godot/build/tests/schola_core_tests
```

Use the matching platform and parallel-build syntax documented in [`Godot/README.md`](Godot/README.md) on macOS or Windows.

### Godot integration tests

Install the pinned GdUnit4 release and run the integration suite using the commands in [`Godot/README.md`](Godot/README.md). CI builds both native extensions, imports the project, loads the example scene, and runs `Godot/tests/integration/` headlessly.

Install test tooling from the plugin root (or from `Resources/python`):

```bash
pip install --group test -e "./Resources/python[all]"
```

### Python tests

From the **plugin root** (directory containing `pytest.ini`):

```bash
python -m pytest Test --import-mode=importlib -n 0
```

Narrow runs with `-k` or test node IDs. Python tests live under `Test/`.

## Documentation

The existing upstream user-facing documentation is built with Doxygen, Sphinx, and Breathe:

1. Install [Doxygen](https://www.doxygen.nl/).
2. `pip install --group docs -e "./Resources/python[all]"`
3. From the plugin root: `schola build-docs --builder html`

The current Doxygen configuration scans the upstream C++ modules under `Source/`; it does not yet publish Godot C++ API documentation. Continue using Doxygen-style comments in Godot C++ so those APIs can join the upstream documentation without a comment-style migration.

Update Sphinx sources under `Docs/Sphinx/` when Python behavior or CLI options change. Update `Godot/README.md` when tool versions or executable development commands change. Update `Godot/docs/development_contract.md` when binding engineering rules change. Preserve course-required project material in the root README and `deliverables/`.

## Building third-party dependencies

Godot dependencies must use the versions and acquisition methods recorded in `Godot/dependencies.lock.json`, `.gitmodules`, and the shared build configuration. Do not select dependency versions independently in a user-story branch. Do not commit updated third-party code or generated artifacts unless the pull request is specifically updating that dependency and records its provenance and integrity information.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project. See [LICENSE.txt](LICENSE.txt).
