#!/bin/bash
# From Apache 2.0 Licensed https://github.com/vizor-games/InfraworldRuntime/blob/master/Setup.sh
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
######################################VARS#############################################################################

# Exit on errors if any
set -e

###############################################################################
# Should be defined as an environment variable; defaults to the pinned repo version otherwise.
branch=${branch:-v1.80.0}
expected_protobuf_version=${expected_protobuf_version:-31.1}
# original branch was v1.23.x
ZLIB_VERSION="1.3"
OPENSSL_VERSION="1.1.1t"

clean=${clean:-true}

VAR_GIT_BRANCH=$branch
VAR_CLEAR_REPO=$clean

REMOTE_ORIGIN="https://github.com/grpc/grpc.git"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GRPC_FOLDER_NAME=grpc
GRPC_ROOT="/tmp/${GRPC_FOLDER_NAME}"
BUILD_DIR="${GRPC_ROOT}/.build"

#will likely neet to install libtool-bin cmake and pkg-config
DEPS=(git automake autoconf libtool cmake strip pkg-config)

# Linux needs an existing UE installation. Default to UE_5.7 in Home directory
UE_ROOT=${UE_ROOT:-"$HOME/UE_5.7"}

if [ ! -d "$UE_ROOT" ]; then
    echo "UE_ROOT directory ${UE_ROOT} does not exist, please set correct UE_ROOT"
    exit 1
fi;

# Compute arch string using uname
UNAME_MACH=$(echo $(uname -m) | tr '[:upper:]' '[:lower:]')
UNAME_OS=$(echo $(uname) | tr '[:upper:]' '[:lower:]')
UNAME_ARCH="${UNAME_MACH}-unknown-${UNAME_OS}-gnu"

# HostLinux Clang lives under .../Linux_x64/<sdk-name>/<triplet>. UE versions ship different sdk-name
# (e.g. v22_clang-16.0.6-centos7 vs v26_clang-20.1.8-rockylinux8).
# Priority: UE_LINUX_CLANG_SDK, then scan Linux_x64 (version-sorted; last match wins), then legacy v22 path.
UE_LINUX_SDK_BASE="${UE_ROOT}/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64"
if [ ! -d "$UE_LINUX_SDK_BASE" ]; then
    echo "UE HostLinux SDK directory does not exist: ${UE_LINUX_SDK_BASE}"
    echo "Install the Linux native toolchain for your UE version (Epic Games Launcher), or set UE_ROOT to a UE install that includes HostLinux SDKs."
    exit 1
fi
if [ -n "${UE_LINUX_CLANG_SDK:-}" ]; then
    UE_CLANG_ROOT="${UE_LINUX_SDK_BASE}/${UE_LINUX_CLANG_SDK}/${UNAME_ARCH}"
else
    UE_CLANG_ROOT=""
    for d in $(ls -1 "$UE_LINUX_SDK_BASE" | sort -V); do
        if [ -x "${UE_LINUX_SDK_BASE}/${d}/${UNAME_ARCH}/bin/clang" ]; then
            UE_CLANG_ROOT="${UE_LINUX_SDK_BASE}/${d}/${UNAME_ARCH}"
        fi
    done
    if [ -z "${UE_CLANG_ROOT}" ] && [ -x "${UE_LINUX_SDK_BASE}/v22_clang-16.0.6-centos7/${UNAME_ARCH}/bin/clang" ]; then
        UE_CLANG_ROOT="${UE_LINUX_SDK_BASE}/v22_clang-16.0.6-centos7/${UNAME_ARCH}"
    fi
fi

if [ -z "${UE_CLANG_ROOT}" ] || [ ! -x "${UE_CLANG_ROOT}/bin/clang" ] || [ ! -x "${UE_CLANG_ROOT}/bin/clang++" ]; then
    echo "Could not find UE HostLinux clang for ${UNAME_ARCH} under ${UE_LINUX_SDK_BASE}"
    echo "Install the Linux native toolchain in Epic Games Launcher, or set UE_LINUX_CLANG_SDK to the folder name (see ls \"${UE_LINUX_SDK_BASE}\")."
    exit 1
fi
echo "UE_CLANG_ROOT=${UE_CLANG_ROOT}"
###############################################################################

OPENSSL_LIB_DIR="${UE_ROOT}/Engine/Source/ThirdParty/OpenSSL/${OPENSSL_VERSION}/lib/Unix/${UNAME_ARCH}"
OPENSSL_INCLUDE="${UE_ROOT}/Engine/Source/ThirdParty/OpenSSL/${OPENSSL_VERSION}/include/Unix"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "GRPC_ROOT=${GRPC_ROOT}"

# Check if all tools are installed
for i in ${DEPS[@]}; do
    if [ ! "$(which ${i})" ];then
       echo "${i} not found, install via 'apt-get install ${i}'" && exit 1
    fi
done

# Check if ran under Linux
if [ $(uname) != 'Linux' ]; then
    echo "Can not work under $(uname) operating system, should be Linux! Exiting..."
    exit 1
fi;

# Clone or pull
if [ ! -d "$GRPC_ROOT" ]; then
    echo "Cloning repo into ${GRPC_ROOT}"
    git clone $REMOTE_ORIGIN $GRPC_ROOT
else
    [[ ${VAR_CLEAR_REPO} ]] && cd $GRPC_ROOT && git merge --abort || true; git clean -fdx && git checkout -f .
    echo "Pulling repo"
    (cd $GRPC_ROOT && git fetch)
fi

echo "Checking out branch ${VAR_GIT_BRANCH}"
(cd $GRPC_ROOT && git checkout tags/$VAR_GIT_BRANCH || true)


if [ "$VAR_GIT_BRANCH" \< "v1.63.0" ]; then
    echo "${VAR_GIT_BRANCH} is less than v1.63.0 patching systemd.cmake"
    (cd $GRPC_ROOT && git checkout tags/v1.63.0 cmake/systemd.cmake || true)
fi

# Update submodules needed for the CMake build. Protobuf comes from the gRPC
# submodule pinned by the selected gRPC release.
(cd $GRPC_ROOT && git submodule update --init --recursive --depth 1 third_party/abseil-cpp third_party/cares/cares third_party/re2 third_party/protobuf third_party/zlib)

if [ "$VAR_CLEAR_REPO" = "true" ]; then
    echo "Cleaning repo and submodules because VAR_CLEAR_REPO is set to ${VAR_CLEAR_REPO}"
    (cd $GRPC_ROOT && make clean)
    (cd $GRPC_ROOT && git clean -fdx)
    (cd $GRPC_ROOT && git submodule foreach git clean -fdx)
elif [ "$VAR_CLEAR_REPO" = "false" ]; then
    echo "Cleaning is not needed!"
else
    echo "Undefined behaviour, VAR_CLEAR_REPO is ${VAR_CLEAR_REPO}!"
    exit 1
fi

PROTOBUF_VERSION_FILE="${GRPC_ROOT}/third_party/protobuf/version.json"
if ! grep -q "\"protoc_version\": \"${expected_protobuf_version}\"" "${PROTOBUF_VERSION_FILE}"; then
    echo "Expected protobuf ${expected_protobuf_version} from gRPC ${VAR_GIT_BRANCH}, but version.json did not match."
    exit 1
fi


HEADERS_DIR="${SCRIPT_DIR}/../../../Source/ThirdParty/gRPC/Linux/include"
LIBS_DIR="${SCRIPT_DIR}/../../../Source/ThirdParty/gRPC/Linux/lib"

# (re)-create headers directory
if [ -d "$HEADERS_DIR" ]; then
    printf '%s\n' "Removing old $HEADERS_DIR"
    rm -rf "$HEADERS_DIR"
fi

if [ -d "$LIBS_DIR" ]; then
    printf '%s\n' "Removing old $LIBS_DIR"
    rm -rf "$LIBS_DIR"
fi

mkdir $HEADERS_DIR -p
mkdir $LIBS_DIR -p


cp -R "${GRPC_ROOT}/include"/* $HEADERS_DIR
cp -R "${GRPC_ROOT}/third_party/protobuf/src/google" "${HEADERS_DIR}"
cp -R "${GRPC_ROOT}/third_party/abseil-cpp/absl" "${HEADERS_DIR}"

export HAS_SYSTEM_CARES=false
export HAS_SYSTEM_PROTOBUF=false
export HAS_SYSTEM_ZLIB=false

# libc++: some UE trees ship LibCxx under Engine/Source; others (e.g. Epic Linux build) only ship it
# inside the HostLinux SDK next to clang. With -nostdinc++, include paths must match that layout.
UE_LIBCXX_INCLUDE="${UE_ROOT}/Engine/Source/ThirdParty/Unix/LibCxx/include"
if [ -d "${UE_LIBCXX_INCLUDE}/c++/v1" ]; then
    LIBCXX_IFLAGS="-I${UE_LIBCXX_INCLUDE} -I${UE_LIBCXX_INCLUDE}/c++/v1"
    LIBCXX_LIB_DIR="${UE_ROOT}/Engine/Source/ThirdParty/Unix/LibCxx/lib/Unix/${UNAME_ARCH}"
elif [ -d "${UE_CLANG_ROOT}/include/c++/v1" ]; then
    LIBCXX_IFLAGS="-I${UE_CLANG_ROOT}/include/c++/v1"
    if [ -f "${UE_CLANG_ROOT}/lib64/libc++.a" ]; then
        LIBCXX_LIB_DIR="${UE_CLANG_ROOT}/lib64"
    elif [ -f "${UE_CLANG_ROOT}/lib/libc++.a" ]; then
        LIBCXX_LIB_DIR="${UE_CLANG_ROOT}/lib"
    else
        echo "libc++.a not found under ${UE_CLANG_ROOT}/lib64 or ${UE_CLANG_ROOT}/lib"
        exit 1
    fi
else
    echo "Could not find libc++ headers (expected ${UE_LIBCXX_INCLUDE}/c++/v1 or ${UE_CLANG_ROOT}/include/c++/v1)."
    exit 1
fi
if [ ! -f "${LIBCXX_LIB_DIR}/libc++.a" ] || [ ! -f "${LIBCXX_LIB_DIR}/libc++abi.a" ]; then
    echo "libc++ static libraries missing under ${LIBCXX_LIB_DIR}"
    exit 1
fi
echo "LIBCXX_LIB_DIR=${LIBCXX_LIB_DIR}"

IGNORED_WARNINGS="-Wno-deprecated-non-prototype -Wno-expansion-to-defined -Wno-error -Wno-unused-command-line-argument"

# CMake does not use LDLIBS; put runtime libs in LDFLAGS / CMAKE_*_LINKER_FLAGS or link lines miss -lc++/-lc++abi.
UE_LINK_FLAGS="-L${LIBCXX_LIB_DIR} -L${OPENSSL_LIB_DIR} -fuse-ld=lld -lc++ -lc++abi -lc -lm -lgcc_s -lgcc"
export LDFLAGS="${UE_LINK_FLAGS}"
export CXXFLAGS="-O2 -std=c++17 -fPIC -nostdinc++ -stdlib=libc++ --sysroot=${UE_CLANG_ROOT} ${IGNORED_WARNINGS} ${LIBCXX_IFLAGS} "
export CFLAGS="-O2 -fPIC --sysroot=${UE_CLANG_ROOT} ${IGNORED_WARNINGS} "

mkdir -p ${BUILD_DIR}

(cd ${BUILD_DIR} && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_C_COMPILER="${UE_CLANG_ROOT}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${UE_CLANG_ROOT}/bin/clang++" \
    -DCMAKE_EXE_LINKER_FLAGS="${UE_LINK_FLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${UE_LINK_FLAGS}" \
    -DCMAKE_MODULE_LINKER_FLAGS="${UE_LINK_FLAGS}" \
    -DgRPC_SSL_PROVIDER=package \
    -DOPENSSL_INCLUDE_DIR=${OPENSSL_INCLUDE} \
    -DOPENSSL_SSL_LIBRARY=${OPENSSL_LIB_DIR}/libssl.a \
    -DOPENSSL_CRYPTO_LIBRARY=${OPENSSL_LIB_DIR}/libcrypto.a \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DRE2_BUILD_TESTING=OFF \
    -DgRPC_USE_SYSTEMD=OFF \
    -DgRPC_PROTOBUF_PROVIDER=module
    )


(cd ${BUILD_DIR} && cmake --build . --config Release --clean-first -j8)

find "${BUILD_DIR}" -name '*.a' -exec cp -f '{}' $LIBS_DIR ";"
echo "Build done! headers are in ${HEADERS_DIR}, and libs are in ${LIBS_DIR}"