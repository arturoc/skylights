# Skylights

Skylights is a command line tool to bake radiance and irradiance maps for image based lighting.

It converts an input environment map in equirectangular format into 3 cubemaps:

- Radiance
- Irradiance
- Environment map to use optionally as a skybox

```
Usage: skylights [OPTIONS] <input-image>

Arguments:
  <input-image>  Environment map to process

Options:
  -x <cubemap-side>       Side of the output cubemaps in pixels [default: 1024]
  -f <output-format>      Output cubemaps format [default: ktx2] [possible values: ktx1, ktx2, dds, png]
  -n <num-samples>        Number of samples per pixel when calculating radiance and irradiance maps [default: 128]
  -m <strength>           Scales the final baked value in the radiance and irradiance maps [default: 1]
  -c <contrast>           Corrects the contrast of the final color [default: 1]
  -b <brightness>         Corrects the brightness of the final color [default: 1]
  -s <saturation>         Corrects the saturation of the final color 0: grayscale and 1: original color [default: 1]
  -u <hue>                Corrects the hue of the final color in degrees. [possible values: 0..360] [default: 0]
  -h, --help              Print help
```

## To install:

- Install rust following the official Rust install guide: https://www.rust-lang.org/tools/install
- Restart the terminal so the search paths are updated.
- Install the latest cmake adding it to the path and a c++ compiler, in windows Visual Studio 2017 or later, or Build Tools for Visual Studio with the Visual C++ option. In linux gcc or clang.
- Use cargo to install the tool:
```
cargo install skylights
```
- Now skylights should be in the path and can be called from anywhere. `skylights --help` shows a help screen.

## To build from source:

- Install rust following the official Rust install guide: https://www.rust-lang.org/tools/install
- Restart the terminal so the search paths are updated.
- Install the latest cmake adding it to the path and a c++ compiler, in windows Visual Studio 2017 or later, or Build Tools for Visual Studio with the Visual C++ option. In linux gcc or clang.
- Download the source from git:
```
git clone https://github.com/arturoc/skylights
```
- Build and run using cargo (add the corresponding parameters after the `--`):
```
cd skylights
cargo run --profile=debug-optimized -- --help
```

Skylights is sponsored by [novorender](https://novorender.com/)