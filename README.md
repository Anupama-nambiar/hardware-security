

# IRON API and MLIR-based AI Engine Toolchain

[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/Xilinx/mlir-aie?cacheSeconds=86400)](https://github.com/Xilinx/mlir-aie/pulls)
[![GitHub Issues](https://img.shields.io/github/issues/Xilinx/mlir-aie/bug?cacheSeconds=86400)](https://github.com/Xilinx/mlir-aie/issues?q=is%3Aopen+is%3Aissue+label%3Abug)
![GitHub Downloads](https://img.shields.io/github/downloads/Xilinx/mlir-aie/latest-wheels/total?color=blue&cacheSeconds=86400)
![GitHub Downloads 2](https://img.shields.io/github/downloads/Xilinx/mlir-aie/latest-wheels-2/total?color=blue&cacheSeconds=86400)
![GitHub Downloads 3](https://img.shields.io/github/downloads/Xilinx/mlir-aie/latest-wheels-3/total?color=blue&cacheSeconds=86400)
![GitHub Contributors](https://img.shields.io/github/contributors/Xilinx/mlir-aie?cacheSeconds=86400)

_Note: Badge values are cached for up to 24 hours (`cacheSeconds=86400`) to reduce load on Shields.io and GitHub, so counts may lag behind real-time activity._
<p align="left">
  <img src="https://github.com/llvm/mlir-www/blob/main/website/static/LogoAssets/logo/PNG/full_color/mlir-identity-03.png" alt="MLIR logo" height="80" />
  <img src="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png" alt="Python logo" height="80" />
  <img src="https://em-content.zobj.net/source/apple/271/mechanical-arm_1f9be.png" alt="Mechanical Arm" height="80" />
</p>

This project emphasizes fast, open-source toolchains for NPU devices including LLVM-based code generation. IRON contains a close-to-metal toolkit that empowers performance engineers to create fast and efficient designs for Ryzen™ AI NPUs powered by AI Engines. It provides Python APIs that enable developers to harness the unique architectural capabilities of AMD’s NPUs. However, this project is not intended to represent an end-to-end compilation flow for all application designs---it is designed to complement, not replace, mainstream NPU tooling for inference like the [AMD Ryzen™ AI Software Platform](https://github.com/amd/RyzenAI-SW/). Targeting researchers and enthusiasts, IRON is designed to unlock the full potential of NPUs for a wide range of workloads, from machine learning to digital signal processing and beyond. This repository includes programming guides and examples demonstrating the APIs. Additionally, the [Peano](https://github.com/Xilinx/llvm-aie) component extends the LLVM framework by adding support for the AI Engine processor as a target architecture, enabling integration with popular compiler frontends such as `clang`. Developers can leverage the [AIE API header library](https://xilinx.github.io/aie_api/topics.html) to implement efficient vectorized AIE core code in C++ that can be compiled by Peano.


# Getting Started for AMD Ryzen™ AI on Linux

These instructions will guide you through everything required for building and executing a program on the Ryzen™ AI NPU, starting from a fresh bare-bones **Ubuntu 24.04** or **Ubuntu 24.10** install.

## Initial Setup

  > Be sure you have the latest BIOS on your laptop or mini-PC that enables the NPU. See [here](#update-bios).

If starting from `Ubuntu 24.04` you may need to update the Linux kernel to 6.11+ by installing the Hardware Enablement (HWE) stack:

  ```bash
  sudo apt update
  sudo apt install --install-recommends linux-generic-hwe-24.04
  sudo reboot
  ```

## Prerequisites

### BIOS Settings:

Turn off SecureBoot (Allows for unsigned drivers to be installed):
   ```BIOS → Security → Secure boot → Disable```

### Install the XDNA™ Driver and XRT

#### Install from upstream packages (Ubuntu 24.04 with Linux 6.17+)

> Ensure your system is running Linux kernel **6.17 or newer** before installing these packages. On Ubuntu 24.04 you can verify this with:
>
> ```bash
> uname -r
> ```
>
> If your kernel is older than 6.17, upgrade it using your distribution's kernel update mechanism or the kernel upgrade steps described in the [Initial Setup](#initial-setup) section above.
>
#### Clone this repository on local machine
#### Navigate to the mlir-aie folder and setup a virtual environment:
   ```bash
   python3 -m venv ironenv
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

Install the XDNA driver and XRT from the AMD PPA:

> The packaged XRT only supports Python 3.12 for `pyxrt`

```bash
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
sudo reboot
```

if pyxrt is not found ensure the python folder being used is from the virtual ironevn and update the path: 
```bash
 >>export PATH=~/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin:~/mlir-aie/ironenv/bin:$PATH
 >>which python
   /home/aba/mlir-aie/ironenv/bin/python
 >>which aie-opt
   /home/aba/mlir-aie/ironenv/bin/aie-opt

```

> Make sure you are in the `render` group to access the NPU:
>
> ```bash
> sudo usermod -aG render $USER
> ```
>
> You may need to logout and log back in after modifying user groups.

> If you are on a different Linux distribution or kernel not supported by the upstream packages, see [Build from source](#alternative-build-xdna-driver-and-xrt-from-source) below.

Verify the NPU device is present:

```bash
xrt-smi examine
```
If xrt-smi command is not found :
```bash
 xrt-smi: command not found
```
Fix the path to be able to use this command : 
```bash
export PATH=/opt/xilinx/xrt/bin:$PATH
xrt-examine 
```


> Once the xrt-smi command is found the bottom of the output you should see:
>  ```
>  Devices present
>  BDF             :  Name
> ------------------------------------
>  [0000:66:00.1]  :  NPU Strix
>  ```
>  Or the name of the NPU in your device. 

### Install IRON and MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    # Python versions 3.10, 3.11, 3.12, 3.13 and 3.14 are currently supported by our wheels
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
    ```

    > **Note:** CMake **3.30 or newer** is required. If your distribution provides an older
    > version, create and activate the Python virtual environment in the setup step below
    > first, then install a newer CMake into that virtual environment:
    >
    > ```bash
    > python3 -m pip install --upgrade cmake
    > ```
    > If it is already installed but not being used yet, again update the path
    ```bash
      export PATH=/opt/cmake-3.30/bin:$PATH 
    ```

## Install IRON for AMD Ryzen™ AI AIE Application Development

1. Navigate to the mlir-aie folder and start the already initiated virtual environment if not in it:
   ```bash
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

1. Install IRON library by installing the `mlir-aie` wheels:

   For installing the `mlir-aie` wheels, there are 3 options. Note that for whichever path you take,
   it is important to sync the `mlir-aie` wheels version, the github repo commit, and the requirements versions. 
   If you install from something other than the latest wheels, make sure 
   you use the repo commit -- and installation instructions -- from that point in time.

   1. **Latest:** For the latest wheels (not necessarily a release):
      ```bash
      # Install IRON library and mlir-aie from the latest wheel
      python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3
      ```

   1. **Latest Release:** Alternatively, you can install the latest released version of `mlir-aie`.
      ```bash
      # Get the latest release version
      latest_tag_with_v=$(curl -s "https://api.github.com/repos/Xilinx/mlir-aie/releases/latest" | jq -r '.tag_name')
      latest_tag="${latest_tag_with_v#v}"

      # Install IRON library and mlir-aie from the latest stable release
      python3 -m pip install mlir_aie==${latest_tag} -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/${latest_tag_with_v}
      git checkout $latest_tag_with_v
      ```


1. Install the Peano compiler (the `llvm-aie` wheels) and dependencies:
   ```bash
   # Install Peano from llvm-aie wheel
   python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

   ```

1. (Optional) Install Python packages required for development and testing:
   ```bash
   # Install Python requirements for development and testing
   python3 -m pip install -r python/requirements_dev.txt

   # This installs the pre-commit hooks defined in .pre-commit-config.yaml
   pre-commit install

   # Install pre-push hooks for formatting (clang-format, black)
   # These run before push to catch formatting issues before CI
   pre-commit install --hook-type pre-push
   ```

1. Setup environment
   ```bash
   source utils/env_setup.sh
   ```


## Build an IRON Design for AIEs in the AMD Ryzen™ AI NPU

For your design of interest, for instance from [programming_examples](../programming_examples/), 2 steps are needed: (i) build the AIE design and then (ii) build the host code.

### Build Device AIE Part

1. Goto the design of interest and run:
   ```bash
   make
   ```

1. Build host code and execute the design:
    ```bash
    make run
    ```

## Learn more about NPU programming with IRON

1. Continue to the [IRON AIE Application Programming Guide](programming_guide)

1. Additional MLIR-AIE documentation is available on the [website](https://xilinx.github.io/mlir-aie/)

1. AIE API header library documentation for single-core AIE programming in C++ is avaiable [here](https://xilinx.github.io/aie_api/topics.html)

1. If you are a university researcher or student and interested in trying these tools on our Ryzen™ AI AUP Cloud systems, please contact the [AMD University Program](mailto:aup@amd.com)

## Optional: Install AIETools

> You may skip the Vitis™ installation step if you intend to only target AMD XDNA™/AIE-ML (AIE2) and AMD XDNA™ 2 (AIE2P) using our open-source single-core compiler [Peano](https://github.com/Xilinx/llvm-aie). Compiling with `xchesscc` is not supported without installing AMD Vitis™ AIE Essentials.

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Access](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab). We will assume you use the installation directory, `/tools/ryzen_ai-1.3.0/vitis_aie_essentials`.

   > This is an early access lounge, you must register and be granted access at this time.

    1. Download VAIML Installer for Linux based compilation: `ryzen_ai-1.3.0ea1.tgz`

    1. Extract the required tools:

       ``` bash
          tar -xzvf ryzen_ai-1.3.0ea1.tgz
          cd ryzen_ai-1.3.0
          mkdir vitis_aie_essentials
          mv vitis_aie_essentials*.whl vitis_aie_essentials
          cd vitis_aie_essentials
          unzip vitis_aie_essentials*.whl
       ```

1. Set up an AI Engine license.

    1. Get a local license for AI Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).

    1. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`:

1. Setup your environment using the following script for Vitis™ for AIETools:

   ```bash
   #!/bin/bash
    #################################################################################
    # Setup Vitis AIE Essentials
    #################################################################################
    export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
    export PATH=$PATH:${AIETOOLS_ROOT}/bin
    export LM_LICENSE_FILE=/opt/Xilinx.lic
   ```

## Alternative: Build XDNA™ Driver and XRT from source

If the [upstream packages](#install-from-upstream-packages-ubuntu-2404) do not support your kernel or distribution, you can build the driver and XRT from source:

1. Execute the scripted build process:

    > This script will install package dependencies, build the xdna-driver and xrt packages, and install them. *These steps require `sudo` access.*

    ```bash
    bash ./utils/build_drivers.sh
    ```

1. Reboot as directed after the script exits.

    ```bash
    sudo reboot
    ```

1. Check that the NPU is working if the device appears with xrt-smi:

   ```bash
   source /opt/xilinx/xrt/setup.sh
   xrt-smi examine
   ```

## Troubleshooting:

### Update BIOS:

Be sure you have the latest BIOS for your laptop or mini PC, this will ensure the NPU (sometimes referred to as IPU) is enabled in the system. You may need to manually enable the NPU:
   ```Advanced → CPU Configuration → IPU```

> **NOTE:** Some manufacturers only provide Windows executables to update the BIOS, please do this before installing Ubuntu.

# Detailed Getting Started Guides and Documentation:

[IRON AIE Application Programming Guide](programming_guide)

[Device Descriptions](docs/Devices.md)

[Building mlir-aie tools from source](docs/Building.md)

[MLIR Dialect and Compiler Documentation](https://xilinx.github.io/mlir-aie/)

Interested in contributing MLIR-AIE? [Information for developers](./CONTRIBUTING.md)

-----

<p align="center">Copyright&copy; 2019-2024 Advanced Micro Devices, Inc</p>
