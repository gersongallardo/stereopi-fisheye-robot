# Python point-cloud sample for Orbbec Astra+

This folder hosts a Python reimplementation of the `Sample-PointCloud` example
shipped with the Orbbec SDK. The script uses Python's built-in `ctypes` module
to call the same C API functions that the original `point_cloud.c` file invokes.
It opens the depth (and, when available, color) streams, configures depth-to-
color alignment, and exports ASCII PLY files for both depth-only and RGBD point
clouds.

## Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Includes the `ctypes` module used for the bindings. |
| Orbbec SDK | 1.10.28 (bundled) | Provides `libOrbbecSDK` as well as the headers used to mirror the structs. |
| System libs | `libudev-dev`, `libusb-1.0-0-dev` (Linux) | Match the dependencies documented in `doc/tutorial/English/Environment_Configuration.md`. |

### Installing the Orbbec runtime libraries

1. Unpack the official SDK release (or the archives stored next to this folder):
   ```bash
   cd Cams/Orbec-Astra-+
   unzip -q lib.zip -d lib
   ```
   This will produce a `lib/linux_x64/libOrbbecSDK.so` tree that matches the
   layout expected by the Python script. For Windows hosts unzip the same
   archive to obtain `win_x64/OrbbecSDK.dll`.
2. Make the shared libraries discoverable:
   * **Linux/macOS** – point `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH`) to the
     folder that contains `libOrbbecSDK.so`:
     ```bash
     export LD_LIBRARY_PATH="$PWD/lib/linux_x64:$LD_LIBRARY_PATH"
     ```
   * **Windows** – append the directory containing `OrbbecSDK.dll` to the
     `PATH` environment variable.
   Alternatively, pass the absolute path to the shared library via the script's
   `--library-path` argument or set the `ORBBEC_LIB_PATH` environment variable.
3. Make sure USB permissions are configured (Linux only) by running the helper
   from the SDK: `sudo ./scripts/install_udev_rules.sh`.

## Usage

From within `Cams/Orbec-Astra-+/python` run:

```bash
python3 sample_pointcloud.py --mode both --output-dir ./captures
```

Key flags:

* `--mode {rgbd|depth|both}` – choose which point clouds to export. `both`
  creates `rgb_points.ply` and `points.ply` sequentially.
* `--timeout` – frame acquisition timeout in milliseconds (default: 100ms).
* `--max-attempts` – how many frames to request before giving up (default: 10).
* `--library-path` – direct path to `libOrbbecSDK.so`/`OrbbecSDK.dll` when it
  is not on the standard loader path.

The script prints progress messages and stores the resulting PLY files in the
selected output directory.
