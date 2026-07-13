# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

import logging
from pathlib import Path

import pytest

from schola.scripts.fabrica import codegen
from schola.scripts.fabrica.codegen import FabricaCodegenData
from schola.scripts.fabrica.codegen_validation import (
    FabricaCodegenValidationError,
    validate_fabrica_codegen_data,
    validate_generated_body,
)

_TEST_ENV_HEADER = """\
#pragma once
#include "Environment/FabricaEnvironment.h"

class SCHOLATEST_API ATestFabricaEnv : public AFabricaEnvironment
{
    GENERATED_BODY()
};
"""


def _write_test_env_header(tmp_path: Path) -> Path:
    header = tmp_path / "Public" / "Environment" / "TestFabricaEnv.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(_TEST_ENV_HEADER, encoding="utf-8")
    return header


def _test_codegen_env(header: Path) -> codegen.CodegenEnv:
    return codegen.CodegenEnv.from_env_header_and_code_gen_folder(header)


def test_write_gen_cpp_roundtrip(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// init line", reward_body="// reward line"),
        create_if_missing=True,
    )
    cpp = env.generated_cpp_path
    text = cpp.read_text(encoding="utf-8")
    assert "void ATestFabricaEnv::FabricaGeneratedInit()" in text
    assert '#include "Environment/TestFabricaEnv.h"' in text
    init, reward = codegen.read_regions(cpp)
    assert "init line" in (init or "")
    assert "reward line" in (reward or "")
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// init2", reward_body="// reward2"),
        create_if_missing=False,
    )
    init2, reward2 = codegen.read_regions(cpp)
    assert "init2" in (init2 or "")
    assert "reward2" in (reward2 or "")


def test_codegen_env_requires_valid_header(tmp_path: Path) -> None:
    header = tmp_path / "Bad.h"
    header.write_text("#pragma once\nclass Foo {};\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AFabricaEnvironment"):
        codegen.CodegenEnv.from_env_header_and_code_gen_folder(header)


def test_write_gen_cpp_regenerates_invalid_file(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    cpp = env.generated_cpp_path
    cpp.parent.mkdir(parents=True, exist_ok=True)
    cpp.write_text(
        f"{codegen.FABRICA_INIT_START}\n// old init\n{codegen.FABRICA_INIT_END}\n\n"
        f"{codegen.FABRICA_REWARD_START}\n// old reward\n{codegen.FABRICA_REWARD_END}\n",
        encoding="utf-8",
    )
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// new init", reward_body="// new reward"),
        create_if_missing=False,
    )
    text = cpp.read_text(encoding="utf-8")
    assert "void ATestFabricaEnv::FabricaGeneratedInit()" in text
    init, reward = codegen.read_regions(cpp)
    assert "old init" not in (init or "")
    assert "old reward" not in (reward or "")
    assert "new init" in (init or "")
    assert "new reward" in (reward or "")


def test_clean_gen_cpp_regions_clears_candidate(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// parent init", reward_body="// parent reward"),
        create_if_missing=True,
    )
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(
            init_body="// candidate init", reward_body="// candidate reward"
        ),
        create_if_missing=False,
    )
    codegen.clean_gen_cpp_regions(env)
    init, reward = codegen.read_regions(env.generated_cpp_path)
    assert (init or "").strip() == ""
    assert (reward or "").strip() == ""


def test_is_valid_gen_cpp_regex(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// init", reward_body="// reward"),
    )
    cpp = env.generated_cpp_path
    valid = cpp.read_text(encoding="utf-8")
    assert codegen._is_valid_gen_cpp(valid, "ATestFabricaEnv")
    assert not codegen._is_valid_gen_cpp(
        f"{codegen.FABRICA_INIT_START}\n{codegen.FABRICA_INIT_END}\n",
        "ATestFabricaEnv",
    )
    assert not codegen._is_valid_gen_cpp(valid, "AOtherEnv")


def test_derive_generated_cpp_path_public_to_private(tmp_path: Path) -> None:
    header = (
        tmp_path
        / "Source"
        / "Module"
        / "Public"
        / "Environment"
        / "ATestEnv.h"
    )
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    gen = codegen.derive_generated_cpp_path(header)
    assert gen == (
        tmp_path / "Source" / "Module" / "Private" / "Environment" / "ATestEnv.fabrica.gen.cpp"
    )


def test_derive_generated_cpp_path_without_public(tmp_path: Path) -> None:
    header = tmp_path / "Environment" / "ATestEnv.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    gen = codegen.derive_generated_cpp_path(header)
    assert gen == tmp_path / "Environment" / "ATestEnv.fabrica.gen.cpp"


def test_derive_generated_cpp_path_code_gen_folder_override(tmp_path: Path) -> None:
    header = (
        tmp_path
        / "Source"
        / "Module"
        / "Public"
        / "Environment"
        / "ATestEnv.h"
    )
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("", encoding="utf-8")
    out_dir = tmp_path / "Custom" / "Gen"
    gen = codegen.derive_generated_cpp_path(header, out_dir)
    assert gen == out_dir / "ATestEnv.fabrica.gen.cpp"

def test_inject_and_clean_env_header_declarations(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    original = header.read_text(encoding="utf-8")
    assert "FabricaGeneratedInit" not in original

    inject = codegen.inject_env_header_declarations(env)
    assert inject.changed
    injected = header.read_text(encoding="utf-8")
    assert codegen.FABRICA_DECL_START in injected
    assert "virtual void FabricaGeneratedInit() override;" in injected
    assert "virtual void FabricaGeneratedRewardForAgent" in injected

    inject_again = codegen.inject_env_header_declarations(env)
    assert not inject_again.changed

    cleanup = codegen.clean_env_header_declarations(header)
    assert cleanup.changed
    assert header.read_text(encoding="utf-8") == original


def test_cleanup_fabrica_run_artifacts_removes_stale_files(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    original = header.read_text(encoding="utf-8")
    cpp = env.generated_cpp_path

    codegen.inject_env_header_declarations(env)
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// stale", reward_body="// stale"),
        create_if_missing=True,
    )
    assert cpp.is_file()
    assert codegen.FABRICA_DECL_START in header.read_text(encoding="utf-8")

    cleanup = codegen.cleanup_fabrica_run_artifacts(
        env_header_path=header,
        gen_path=cpp,
    )
    assert cleanup.changed
    assert cleanup.header_changed
    assert cleanup.gen_cpp_changed
    assert header.read_text(encoding="utf-8") == original
    assert not cpp.is_file()


def test_cleanup_fabrica_run_artifacts_removes_gen_cpp_when_header_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    header = _write_test_env_header(tmp_path)
    cpp = tmp_path / "TestFabricaEnv.fabrica.gen.cpp"
    cpp.write_text("// stale generated file\n", encoding="utf-8")

    def _fail_header_cleanup(_header: Path) -> codegen.FabricaCodegenResult:
        raise OSError("header locked")

    monkeypatch.setattr(codegen, "clean_env_header_declarations", _fail_header_cleanup)

    with caplog.at_level(logging.WARNING):
        cleanup = codegen.cleanup_fabrica_run_artifacts(
            env_header_path=header,
            gen_path=cpp,
        )

    assert not cleanup.header_changed
    assert cleanup.gen_cpp_changed
    assert cleanup.changed
    assert not cpp.is_file()
    assert any("header locked" in record.message for record in caplog.records)


def test_inject_env_header_declarations_after_generated_body(tmp_path: Path) -> None:
    header = tmp_path / "Public" / "Environment" / "TestFabricaEnv.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        """\
#pragma once
#include "Environment/FabricaEnvironment.h"

class SCHOLATEST_API ATestFabricaEnv : public AFabricaEnvironment
{
    GENERATED_BODY()

protected:
    virtual void OnUserReset_Implementation(FInitialAgentState& OutAgentState) override;

private:
    float Value = 0.f;
};
""",
        encoding="utf-8",
    )
    codegen.inject_env_header_declarations(_test_codegen_env(header))
    text = header.read_text(encoding="utf-8")
    gen_idx = text.index("GENERATED_BODY()")
    protected_idx = text.index("protected:")
    decl_idx = text.index(codegen.FABRICA_DECL_START)
    assert gen_idx < decl_idx < protected_idx
    assert text[gen_idx + len("GENERATED_BODY()") : decl_idx].strip() == ""


def test_validate_generated_body_rejects_preprocessor_directives() -> None:
    with pytest.raises(FabricaCodegenValidationError, match="preprocessor directive"):
        validate_generated_body('#include "Engine/Engine.h"\n', region="init_body")
    with pytest.raises(FabricaCodegenValidationError, match="preprocessor directive"):
        validate_generated_body("#pragma once\n", region="reward_body")


@pytest.mark.parametrize(
    "body",
    [
        "FPlatformProcess::CreateProc(TEXT(\"cmd\"), nullptr, true, false, false, nullptr, 0, nullptr, nullptr);",
        "IFileManager::Get().Delete(TEXT(\"foo.txt\"));",
        "FFileHelper::SaveStringToFile(TEXT(\"x\"), TEXT(\"out.txt\"));",
        "FSocket* Socket = ISocketSubsystem::Get()->CreateSocket(NAME_Stream, TEXT(\"x\"), false);",
        "system(\"rm -rf /\");",
        "FILE* F = fopen(\"/etc/passwd\", \"r\");",
        "void* Lib = LoadLibrary(TEXT(\"kernel32.dll\"));",
        "FILE* Pipe = popen(\"ls\", \"r\");",
    ],
)
def test_validate_generated_body_rejects_denylisted_tokens(body: str) -> None:
    with pytest.raises(FabricaCodegenValidationError, match="disallowed token"):
        validate_generated_body(body, region="reward_body")


def test_validate_generated_body_allows_empty_body() -> None:
    validate_generated_body("", region="init_body")
    validate_generated_body("   \n\t  ", region="reward_body")


def test_validate_generated_body_allows_safe_reward_code() -> None:
    validate_generated_body(
        'FabricaTrackedActors.Add(TEXT("Agent"), AgentActor);',
        region="init_body",
    )
    validate_generated_body(
        'RewardComponents.Add(TEXT("fabrica_r:progress"), FString::SanitizeFloat(1.f));',
        region="reward_body",
    )


def test_validate_fabrica_codegen_data_rejects_unsafe_code() -> None:
    with pytest.raises(FabricaCodegenValidationError, match="preprocessor directive"):
        validate_fabrica_codegen_data(
            FabricaCodegenData(
                init_body="",
                reward_body='#include "Malicious.h"\n',
            )
        )


def test_clean_gen_cpp_regions_allows_empty_bodies(tmp_path: Path) -> None:
    header = _write_test_env_header(tmp_path)
    env = _test_codegen_env(header)
    codegen.write_gen_cpp(
        env,
        FabricaCodegenData(init_body="// init", reward_body="// reward"),
        create_if_missing=True,
    )
    validate_fabrica_codegen_data(FabricaCodegenData(init_body="", reward_body=""))
    codegen.clean_gen_cpp_regions(env)
