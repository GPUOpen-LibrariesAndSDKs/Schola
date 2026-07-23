# Contributing to Schola

Thank you for your interest in improving Schola. This guide covers how to set up a development environment, follow project conventions, and submit changes.

For installation, engine versions, and dependency extras, see the [README](README.md).

## Ways to contribute

- **Bug reports** — Use the [bug report](.github/ISSUE_TEMPLATE/bug_report.yml) template and include reproduction steps and system information when you can.
- **Features and improvements** — Use the [feature or improvement](.github/ISSUE_TEMPLATE/feature_request.yml) template. You do not need to frame every idea as fixing a specific problem.
- **Pull requests** — Code, tests, and documentation fixes are welcome via PR. Fill out the [pull request template](.github/pull_request_template.md) when you open a PR.
- **Documentation** — User-facing changes should update the README and/or Sphinx guides under `Docs/Sphinx/` as appropriate.

Discuss larger changes in an issue first if you are unsure about direction or scope; that helps avoid rework. For questions that are not a good fit for a public GitHub issue, contact the Schola team at [schola@amd.com](mailto:schola@amd.com).

## Development setup

1. Copy or clone this repository into your Unreal project’s `Plugins` folder (see [Installing Schola Into Your Project](README.md#installing-schola-into-your-project) in the README).
2. Install a supported Unreal Engine version for your Schola release (see the compatibility table in the README).
3. Install the Python package in editable mode from the plugin root:

   ```bash
   pip install --group test -e "./Resources/python[all]"
   ```

4. Recompile your Unreal project after adding or updating native plugin code.

Optional extras (`sb3`, `rllib`, `minari`, `docs`) are described in the README and in `Resources/python/pyproject.toml`.

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
| `chore` | Maintenance (tooling, configs) that does not fit above |

Optional scopes help when a change is localized, for example `python`, `unreal`, `proto`, or `docs`.

Examples:

```text
fix(python): handle empty observation buffers in rollout worker
feat(unreal): add StateTree evaluator for policy actions
docs: document pytest unreal_timeout
test: add coverage for Minari dataset export
```

Use the imperative mood in the subject line (“add feature”, not “added feature”). Add a body after a blank line when context, trade-offs, or breaking changes need explanation. Breaking changes can be called out with a `BREAKING CHANGE:` footer as described in the Conventional Commits spec.

## Coding standards

### Unreal C++

- Follow the [Unreal Engine coding standard](https://docs.unrealengine.com/4.27/en-US/ProductionPipelines/DevelopmentSetup/CodingStandard/).
- Use Doxygen-style comments (`/** ... */`) for non-obvious APIs. In Visual Studio you can switch C++ comment style to Doxygen under **Tools → Options → Text Editor → C/C++ → Code Style → General**.
- [UE-Clang-Format](https://github.com/TensorWorks/UE-Clang-Format) is a useful optional formatter for Visual Studio.

Place new automation tests under `Source/<Module>/Private/Test/` as `*Test.cpp` files using `IMPLEMENT_SIMPLE_AUTOMATION_TEST`.

### Python

- Format with [Black](https://black.readthedocs.io/). Run Black locally before pushing.
- Follow PEP 8 aside from Black’s formatting choices.
- Use [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for public APIs. Sphinx-compatible RST in docstrings is supported.

### Copyright headers

New source files should include the standard AMD copyright notice at the top, matching neighboring files:

- C++: `// Copyright (c) <year> Advanced Micro Devices, Inc. All Rights Reserved.`
- Python: `# Copyright (c) <year> Advanced Micro Devices, Inc. All Rights Reserved.`


### Protocol buffers and generated code

If you change `.proto` definitions, regenerate bindings from the plugin root:

```bash
schola compile-proto
```

Do not hand-edit generated `schola/generated` Python modules beyond what is done in the compile-proto script.

## Testing
Schola includes unit tests for both Python and C++ source. The primary workflow is **pytest**, including for C++ Unreal automation tests. Python tests live under `/Test`. C++ tests are **Unreal automation tests** in `Source/<Module>/Private/Test` (files named `*Test.cpp` using `IMPLEMENT_SIMPLE_AUTOMATION_TEST`). The same tests can be run from the Unreal editor via the standard workflow for running Unreal Automation Tests.

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

### Unreal C++ automation tests

C++ tests are collected from `Source/**/Private/Test/*Test.cpp` as Unreal Automation Tests. Pytest can launch them on **Windows** only using a custom pytest adapter; on Linux tests must be run from the editor. 

#### Running Unreal C++ automation tests via pytest

Collection and execution of `*Test.cpp` automation tests is implemented in `Source/conftest.py` and **only runs on Windows** (`win32`). On Linux, those files are ignored so pytest does not hang waiting for the editor.

Requirements:

- A **host `.uproject`** that references your engine version. Pytest searches upward from the config root for a `.uproject` (for example next to `Plugins/Schola` in a normal game project).
- The **Unreal Editor** for that engine, resolved from `engine_path` / `--engine-path` (see `pytest.ini` or CLI).
- Optional **one-step build** before tests: `build_unreal = true` in `pytest.ini`, or pass `--build-unreal`, or skip with `--no-build-unreal`.

Typical invocation (plugin embedded under `Plugins/Schola`, tests only under `Source`):

```bash
python -m pytest ./Source --rootdir=./Source --import-mode=importlib -n 0 \
  -o engine_path="C:/Program Files/Epic Games/UE_5.7" -o build_unreal=true
```

Useful options and settings:

| Item | Purpose |
| ---- | ------- |
| `engine_path` / `--engine-path` | Root of the Unreal Engine install (contains `Engine/`). |
| `build_unreal` / `--build-unreal` / `--no-build-unreal` | Whether to build the game project through UBT before launching the editor. |
| `unreal_timeout` / `--unreal-timeout` | Global timeout in seconds for all Unreal batches (default `240`; CLI overrides ini, e.g. `-o unreal_timeout=300` or `--unreal-timeout 300`). |

Session artifacts (JSON reports per batch, command-line files, coverage outputs) are written under `.pytest_tmp` (see `addopts` in `pytest.ini`).

#### C++ code coverage (OpenCppCoverage) via pytest

On **Windows**, pytest can drive **[OpenCppCoverage](https://github.com/OpenCppCoverage/OpenCppCoverage)** so Unreal editor runs are instrumented and merged into HTML and/or Cobertura XML for **Schola plugin** sources and binaries.

1. Install OpenCppCoverage and ensure `OpenCppCoverage.exe` is on `PATH`, or set **`SCHOLA_OPENCPPCOVERAGE`** to the full path of the executable.
2. Enable exports using **either** the command line **or** `pytest.ini`:

   - `--schola-cpp-coverage-html` and/or `--schola-cpp-coverage-cobertura`
   - or `schola_cpp_coverage_html = true` / `schola_cpp_coverage_cobertura = true` in `pytest.ini`

3. Run a **build with coverage enabled** at least once when you change native code: keep `build_unreal` true (or pass `--build-unreal`). While building, pytest sets **`SCHOLA_MEASURE_CPP_COVERAGE=1`**, which Schola `*.Build.cs` files use to disable aggressive optimizations so line coverage is meaningful.

After the run, merged reports are written next to the session Unreal report directory, for example:

- HTML: `.pytest_tmp/unreal_test_reports0/html/`
- Cobertura: `.pytest_tmp/unreal_test_reports0/schola_cpp_coverage.xml`

Per-batch binary `.cov` files may appear under `batch_*` subfolders before the merge step. Optional: **`SCHOLA_OPENCPPCOVERAGE_MERGE_TIMEOUT`** (seconds, default `120`) controls the merge subprocess timeout.

**Note:** `pytest-cov` in `pyproject.toml` is for **Python** coverage. Schola C++ coverage is separate and controlled by the `schola_cpp_coverage_*` options above.

## Documentation

User-facing documentation is built with Doxygen, Sphinx, and Breathe:

1. Install [Doxygen](https://www.doxygen.nl/).
2. `pip install --group docs -e "./Resources/python[all]"`
3. From the plugin root: `schola build-docs --builder html`

Update Sphinx sources under `Docs/Sphinx/` when behavior or CLI options change.

## Building third-party dependencies

Schola ships prebuilt third-party libraries under `Source/ThirdParty`. Rebuild them only if you hit setup issues, using `Resources/Build/windows_dependencies.bat` or `Resources/Build/linux_dependencies.sh` as described in the README. Don't commit updated third party dependencies unless part of a specific commit to update these dependencies.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project. See [LICENSE.txt](LICENSE.txt).