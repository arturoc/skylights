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

Skylights is sponsored by [novorender](https://novorender.com/)