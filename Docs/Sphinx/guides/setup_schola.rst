Getting Started with Schola
===========================

.. include:: ./doc_fragments/prerequisites.rst

Install Schola
--------------

1. Add `Schola` to a project as an Unreal Engine Plugin, either by downloading the source, or via git.

   .. tabs::

      .. tab:: Download and Install 
         
         1. Create a folder named ``Schola`` in the ``Plugins`` folder of your Unreal Engine project.
         2. Download the zipped source code from the `Schola Repository <https://github.com/GPUOpen-LibrariesAndSDKs/Schola>`__
         3. Unzip the repository. Copy the contents of the ``Schola`` folder to the ``Schola`` folder you created in your project.

      .. code-tab:: bash Git Submodule 
         
         git submodule add https://github.com/GPUOpen-LibrariesAndSDKs/Schola.git  ./Plugins/Schola

      .. code-tab:: bash Standalone Git Repository
         
         git clone https://github.com/GPUOpen-LibrariesAndSDKs/Schola.git ./Plugins/Schola

   .. note::
      If you experience an error installing the repository with git due to a large file sizes, run the following command to increase the git buffer size:
      
      .. code-block:: bash
         
         git config --global http.postBuffer 524288000

2. 
   .. include:: ./doc_fragments/pip_install.rst

3. Using an editor of your choice, such as `Visual Studio <https://visualstudio.microsoft.com/vs/>`__ or `Visual Studio Code <https://code.visualstudio.com/download>`__ compile the project source. The Schola plugin will not appear in the editor, or be useable until the project has been compiled.
4. Launch your project as normal in Unreal Engine. You should see the Schola plugin in the Plugins section of the project settings.

Building 3rd-Party Dependencies for Schola (Optional)
-----------------------------------------------------

Schola bundles its gRPC and Protobuf dependencies, and the IREE runtime used by ``ScholaIREE`` on Windows, so Windows users
don't need to build anything. Linux users must build the IREE runtime from a local IREE build (see below). You may also build
any of these dependencies from source if you want a different version of a dependency or Unreal Engine.


Building gRPC from source for Schola
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install Prerequisites

   .. tabs::

      .. group-tab:: Linux

         - |ubuntu_version|_. |ubuntu_version_exact|
         - |unreal_version_linux|_. |unreal_version|
         - |py_version|_
   
         .. note::
            While other Integrated Development Environments (IDEs) can be used, `Visual Studio Code <https://code.visualstudio.com/download>`_ is recommended along with the `C/C++ Extension Pack <https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools>`_. Please see the `Unreal Engine documentation <https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-visual-studio-code-for-unreal-engine>`_ for more information. 
      
      .. group-tab:: Windows
         
         -  `Git VCS <https://git-scm.com/download/win>`_
         -  |unreal_version_win|_
         -  `Visual Studio <https://visualstudio.microsoft.com/downloads/>`_ (2017 with VC++ tools v141 installed, or Visual Studio 2022 with VC++ tools)
         -  `CMake <https://cmake.org/>`_ (used to generate a Visual Studio solution from the ``CMakeLists.txt`` provided with gRPC)
         -  `Strawberry perl <http://strawberryperl.com/>`_ 
         -  `NASM <https://www.nasm.us/>`_

         .. note::
            Some install steps may require adding the above prerequisites to your PATH.

2. Build gRPC using the scripts in `/Resources/Build/grpc`, by running the following from the Plugin Root Dir

   .. tabs::

      .. code-tab:: bash Linux
            
         # Install Required Packages for Running the Build Script
         sudo apt update
         sudo apt-get -y install cmake git automake autoconf libtool-bin pkg-config build-essential

         # Change this to your Unreal Engine path
         UE_ROOT="~/UnrealEngine/UE_5.5"
         export UE_ROOT
         bash ./Resources/Build/grpc/build_grpc_linux.sh
      
      .. note::
         You may need to manually update ``libtool-bin`` as old versions of ``libtool-bin`` may cause errors during install.

      .. code-tab:: batch Windows

         # Change this to your Unreal Engine path if it is different than the default.
         set UE_ROOT="C:\Program Files\Epic Games\UE_5.5"
         ./Resources/Build/grpc/build_grpc_windows.bat

Building the IREE runtime from source for ScholaIREE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Required on Linux. On Windows, the bundled runtime is used unless you want a different IREE version.

1. Build the IREE runtime (v3.14.0, to match the ``iree-base-compiler`` used for model export) by following the steps in
   ``Source/ThirdParty/IREE/README.md``.

2. Copy the built runtime into Schola by running the script for your platform from the plugin root. The paths are your
   IREE install, source, and build directories.

   .. tabs::

      .. code-tab:: powershell Windows

         .\Resources\Build\iree\build_iree_windows.ps1 -IreeInstallRoot C:\iree\iree-install -IreeSourceRoot C:\iree\iree-src -IreeBuildRoot C:\iree\iree-build

      .. code-tab:: bash Linux

         bash ./Resources/Build/iree/build_iree_linux.sh ~/iree/iree-install ~/iree/iree-src ~/iree/iree-build

3. Rebuild Schola. The Unreal Build Tool output should report that the IREE runtime was found.

Regenerated Protobuf and gRPC Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Follow the steps in the `Building Schola from source` section to build gRPC from source, and get an up to date version of `protoc.exe`.

2. Run the following command to regenerate the protobuf and gRPC code.

   .. code-block:: bash

      schola compile-proto --plugin-folder .

.. note::

   Type stub generation (``.pyi`` files) is enabled by default. Pass ``--no-add-type-stubs`` if you need to disable it. Stubs help IDEs with completion, but not every protobuf or gRPC feature is represented cleanly in ``.pyi`` output; if generation fails, try again with stubs disabled.

   You can also run the same logic without the console script: ``python -m schola.scripts.utils.compile_proto`` (same arguments as ``schola compile-proto``).



