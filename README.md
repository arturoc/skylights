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
  -s <cubemap-side>      Side of the output cubemaps in pixels [default: 1024]
  -h, --help             Print help
```

Skylights is sponsored by [novorender](https://novorender.com/)