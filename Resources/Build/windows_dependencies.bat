@echo off
:: From Apache 2.0 Licensed https://github.com/vizor-games/InfraworldRuntime/blob/master/Setup.bat
:: Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
::#####################################VARS#############################################################################
set SCRIPT_FOLDER=%~dp0

set GRPC_ROOT=%TEMP%\grpc
set GRPC_INCLUDE_DIR=%SCRIPT_FOLDER%\..\..\Source\ThirdParty\gRPC\Win64\include
set GRPC_LIBRARIES_DIR=%SCRIPT_FOLDER%\..\..\Source\ThirdParty\gRPC\Win64\lib
set GRPC_PROGRAMS_DIR=%SCRIPT_FOLDER%\..\tools

set CMAKE_BUILD_DIR=%GRPC_ROOT%\.build
set CMAKE_EXE=
set REMOTE_ORIGIN=https://github.com/grpc/grpc.git
set BRANCH=v1.80.0
set EXPECTED_PROTOBUF_VERSION=31.1

set ZLIB_VERSION=1.3
set OPENSSL_VERSION=1.1.1t

set CLEAN_GRPC=0

:: original branch was v1.23.x
::#####################################VARS#############################################################################

:MAIN
    echo ">>>>>>>>>> cleaning"
    CALL :CLEAN
    IF ERRORLEVEL 1 GOTO :ABORT

    IF %CLEAN_GRPC% == 0 (echo ">>>>>>>>>> clone git")
    IF %CLEAN_GRPC% == 0 (CALL :UPDATE_GRPC)
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> resolving unreal root"
    CALL :RESOLVE_UE_ROOT
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> resolving cmake"
    CALL :RESOLVE_CMAKE
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> verifying protobuf version"
    CALL :VERIFY_PROTOBUF_VERSION
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> making build configs"
    CALL :CMAKE
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> building"
    CALL :BUILD
    IF ERRORLEVEL 1 GOTO :ABORT

    echo ">>>>>>>>>> copying files"
    CALL :COPY_ALL
    IF ERRORLEVEL 1 GOTO :ABORT

:GRACEFULEXIT
    CALL :CLEANUP
    echo Build done!
    goto :eof

:ABORT
    echo Aborted...
    CALL :CLEANUP
    goto :eof

:RESOLVE_UE_ROOT
    IF "%UE_ROOT%" == "" set UE_ROOT=C:\Program Files\Epic Games\UE_5.7
    IF NOT EXIST "%UE_ROOT%" (echo "UE_ROOT directory %UE_ROOT% does not exist, please set correct UE_ROOT via SET UE_ROOT=PATH_TO_UNREAL_ENGINE_FOLDER" && EXIT /B 1)
EXIT /B 0

:RESOLVE_CMAKE
    where cmake >NUL 2>&1
    IF NOT ERRORLEVEL 1 (
        set CMAKE_EXE=cmake
        EXIT /B 0
    )
    IF EXIST "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
        set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
        EXIT /B 0
    )
    IF EXIST "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
        set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
        EXIT /B 0
    )
    IF EXIST "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
        set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
        EXIT /B 0
    )
    IF EXIST "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
        set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
        EXIT /B 0
    )
    echo "Could not find cmake.exe on PATH or in Visual Studio 2022. Please install CMake or launch from a Developer Prompt with CMake available."
EXIT /B 1

:VERIFY_PROTOBUF_VERSION
    findstr /C:"\"protoc_version\": \"%EXPECTED_PROTOBUF_VERSION%\"" "%GRPC_ROOT%\third_party\protobuf\version.json" >NUL
    IF NOT ERRORLEVEL 1 EXIT /B 0
    echo "Expected protobuf %EXPECTED_PROTOBUF_VERSION% from gRPC %BRANCH%, but version.json did not match."
EXIT /B 1

:CLEAN
    IF EXIST "%CMAKE_BUILD_DIR%" (rmdir "%CMAKE_BUILD_DIR%" /s /q)
    IF EXIST "%GRPC_INCLUDE_DIR%" (rmdir "%GRPC_INCLUDE_DIR%" /s /q)
    IF EXIST "%GRPC_LIBRARIES_DIR%" (rmdir "%GRPC_LIBRARIES_DIR%" /s /q)
    IF EXIST "%GRPC_PROGRAMS_DIR%" (rmdir "%GRPC_PROGRAMS_DIR%" /s /q)
EXIT /B %ERRORLEVEL%

:UPDATE_GRPC
    IF EXIST "%GRPC_ROOT%" (rmdir "%GRPC_ROOT%" /s /q)
    (call git clone --branch %BRANCH% --depth 1 "%REMOTE_ORIGIN%" "%GRPC_ROOT%" && cd "%GRPC_ROOT%") || EXIT /B %ERRORLEVEL%
    git submodule update --init --recursive --depth 1 third_party/abseil-cpp third_party/cares/cares third_party/re2 third_party/protobuf
EXIT /B %ERRORLEVEL%

:: ABSL, CARES and RE2 building as modules are new additions
:CMAKE
    if NOT EXIST "%CMAKE_BUILD_DIR%" (mkdir "%CMAKE_BUILD_DIR%")
    pushd "%CMAKE_BUILD_DIR%"
    call "%CMAKE_EXE%" .. -G "Visual Studio 17 2022" -A x64 ^
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
        -DCMAKE_CXX_STANDARD=20 ^
        -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
        -DABSL_PROPAGATE_CXX_STD=ON ^
        -DCMAKE_CXX_STANDARD_LIBRARIES="Crypt32.Lib User32.lib Advapi32.lib Shell32.lib" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CONFIGURATION_TYPES=Release ^
        -Dprotobuf_BUILD_TESTS=OFF ^
        -DgRPC_BUILD_TESTS=OFF ^
        -DgRPC_BUILD_CSHARP_EXT=OFF ^
        -DgRPC_ABSL_PROVIDER=module ^
        -DgRPC_CARES_PROVIDER=module ^
        -DgRPC_RE2_PROVIDER=module ^
        -DgRPC_ZLIB_PROVIDER=package ^
        -DZLIB_INCLUDE_DIR="%UE_ROOT%\Engine\Source\ThirdParty\zlib\%ZLIB_VERSION%\include" ^
        -DZLIB_LIBRARY_DEBUG="%UE_ROOT%\Engine\Source\ThirdParty\zlib\%ZLIB_VERSION%\lib\Win64\Release\zlibstatic.lib" ^
        -DZLIB_LIBRARY_RELEASE="%UE_ROOT%\Engine\Source\ThirdParty\zlib\%ZLIB_VERSION%\lib\Win64\Release\zlibstatic.lib" ^
        -DgRPC_SSL_PROVIDER=package ^
        -DLIB_EAY_LIBRARY_DEBUG="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Debug\libcrypto.lib" ^
        -DLIB_EAY_LIBRARY_RELEASE="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libcrypto.lib" ^
        -DLIB_EAY_DEBUG="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Debug\libcrypto.lib" ^
        -DLIB_EAY_RELEASE="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libcrypto.lib" ^
        -DOPENSSL_INCLUDE_DIR="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\include\Win64\VS2015" ^
        -DSSL_EAY_DEBUG="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libssl.lib" ^
        -DSSL_EAY_LIBRARY_DEBUG="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libssl.lib" ^
        -DSSL_EAY_LIBRARY_RELEASE="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libssl.lib" ^
        -DSSL_EAY_RELEASE="%UE_ROOT%\Engine\Source\ThirdParty\OpenSSL\%OPENSSL_VERSION%\Lib\Win64\VS2015\Release\libssl.lib" ^
        -Dprotobuf_DISABLE_RTTI=ON ^
        -DgRPC_PROTOBUF_PROVIDER=module
    popd
EXIT /B %ERRORLEVEL%

:BUILD
    pushd "%CMAKE_BUILD_DIR%"
    call "%CMAKE_EXE%" --build . --config Release --clean-first -j8
    popd
EXIT /B %ERRORLEVEL%

:COPY_ALL
    CALL :COPY_HEADERS
    IF ERRORLEVEL 8 EXIT /B %ERRORLEVEL%
    CALL :COPY_LIBRARIES
    IF ERRORLEVEL 1 EXIT /B %ERRORLEVEL%
    CALL :COPY_PROGRAMS
EXIT /B %ERRORLEVEL%

:COPY_HEADERS
    echo ">>>>>>>>>> copy headers"
    robocopy "%GRPC_ROOT%\include" "%GRPC_INCLUDE_DIR%" /E
    robocopy "%GRPC_ROOT%\third_party\protobuf\src\google" "%GRPC_INCLUDE_DIR%\google" /E
    robocopy "%GRPC_ROOT%\third_party\abseil-cpp\absl" "%GRPC_INCLUDE_DIR%\absl" /E
EXIT /B %ERRORLEVEL%

:COPY_LIBRARIES
    echo ">>>>>>>>>> copy libraries"
    if NOT EXIST "%GRPC_LIBRARIES_DIR%" (mkdir "%GRPC_LIBRARIES_DIR%")
    for /R "%CMAKE_BUILD_DIR%" %%f in (*.lib) do copy "%%f" "%GRPC_LIBRARIES_DIR%\"
    dir /b "%GRPC_LIBRARIES_DIR%\*.lib" >NUL 2>&1 || (echo "Failed to copy .lib outputs into %GRPC_LIBRARIES_DIR%" && EXIT /B 1)
EXIT /B %ERRORLEVEL%

:COPY_PROGRAMS
    echo ">>>>>>>>>> copy programs"
    if NOT EXIST "%GRPC_PROGRAMS_DIR%" (mkdir "%GRPC_PROGRAMS_DIR%")
    for /R "%CMAKE_BUILD_DIR%" %%f in (*.exe) do copy "%%f" "%GRPC_PROGRAMS_DIR%\"
    IF NOT EXIST "%GRPC_PROGRAMS_DIR%\protoc.exe" (echo "Failed to copy protoc.exe" && EXIT /B 1)
    IF NOT EXIST "%GRPC_PROGRAMS_DIR%\grpc_cpp_plugin.exe" (echo "Failed to copy grpc_cpp_plugin.exe" && EXIT /B 1)
    IF NOT EXIST "%GRPC_PROGRAMS_DIR%\grpc_python_plugin.exe" (echo "Failed to copy grpc_python_plugin.exe" && EXIT /B 1)
EXIT /B %ERRORLEVEL%

:REMOVE_USELESS_LIBRARIES
    echo ">>>>>>>>>> remove useless libraries"
    ::del "%GRPC_LIBRARIES_DIR%\grpc_csharp_ext.lib"
    ::del "%GRPC_LIBRARIES_DIR%\gflags_static.lib"
    ::del "%GRPC_LIBRARIES_DIR%\gflags_nothreads_static.lib"
    ::del "%GRPC_LIBRARIES_DIR%\benchmark.lib"
goto :eof

:CLEANUP
    echo  ">>>>>>>>>> Cleaning Up"
    cd "%SCRIPT_FOLDER%"
EXIT /B %ERRORLEVEL%
