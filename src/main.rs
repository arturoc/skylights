use std::{borrow::Cow, error::Error, mem::size_of, ptr, ffi::CString, cell::OnceCell};
use bytemuck::{Pod, Zeroable};
use half::f16;
use image::{DynamicImage, ImageBuffer, Rgb};
use libktx_rs_sys::{ktxTexture2_Create, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, ktxTexture1_Create, ktxTexture};
use num_traits::FromPrimitive;
use wgpu::{util::{DeviceExt, BufferInitDescriptor}, TextureDescriptor, TextureFormat, TextureUsages, Origin3d, ImageDataLayout, ImageCopyTexture};
use async_std::{prelude::*, fs::File, task::spawn_blocking, path::Path};
use anyhow::{Context, Result};

static RE_CONSTANTS: &str = r"const[ \t]+([A-Z][A-Z0-9_]*)[ \t]*(:)?[ \t]*([^ \t=]+)?[ \t]*=[ \t]*([^ \t;]*);";
const CONST_RE: OnceCell<regex::Regex> = OnceCell::new();

static RE_TEXTURE_STORAGE: &str = r"var ([a-z0-9_]+): texture_storage_([^<]+)[ \t]*<[ \t]*([^,]*)[ \t]*,[ \t]*(read|write)[ \t]*>;";
const TEXTURE_STORAGE_RE: OnceCell<regex::Regex> = OnceCell::new();

static RE_TEXTURE_CUBE: &str = r"var ([a-z0-9_]+): texture_cube[ \t]*<[ \t]*([^>]*)[ \t]*>;";
const TEXTURE_CUBE_RE: OnceCell<regex::Regex> = OnceCell::new();

#[test]
fn test_texture_storage_re() {
    let captures = TEXTURE_STORAGE_RE
        .get_or_init(|| regex::Regex::new(RE_TEXTURE_STORAGE).unwrap())
        .captures("var lut: texture_storage_2d<rgba32float, write>;");
    assert!(captures.is_some());
    let captures = captures.as_ref().unwrap();
    assert_eq!(&captures[1], "lut");
    assert_eq!(&captures[2], "2d");
    assert_eq!(&captures[3], "rgba32float");
    assert_eq!(&captures[4], "write");


    let captures = TEXTURE_STORAGE_RE
        .get_or_init(|| regex::Regex::new(RE_TEXTURE_STORAGE).unwrap())
        .captures("var lut: texture_storage_2d_array<rgba32float, read>;");
    assert!(captures.is_some());
    let captures = captures.as_ref().unwrap();
    assert_eq!(&captures[1], "lut");
    assert_eq!(&captures[2], "2d_array");
    assert_eq!(&captures[3], "rgba32float");
    assert_eq!(&captures[4], "read");
}

#[test]
fn test_texture_cube_re() {
    let captures = TEXTURE_CUBE_RE
        .get_or_init(|| regex::Regex::new(RE_TEXTURE_CUBE).unwrap())
        .captures("var envmap: texture_cube<f32>;");
    assert!(captures.is_some());
    let captures = captures.as_ref().unwrap();
    assert_eq!(&captures[1], "envmap");
    assert_eq!(&captures[2], "f32");
}


#[test]
fn test_constant_re() {
    let captures = CONST_RE
        .get_or_init(|| regex::Regex::new(RE_CONSTANTS).unwrap())
        .captures("const ENVIRONMENT_SCALE: f32 = 2.0;");
    assert!(captures.is_some());
    let captures = captures.as_ref().unwrap();
    assert_eq!(&captures[1], "ENVIRONMENT_SCALE");
    assert_eq!(&captures[2], ":");
    assert_eq!(&captures[3], "f32");
    assert_eq!(&captures[4], "2.0");
    let captures = CONST_RE
        .get_or_init(|| regex::Regex::new(RE_CONSTANTS).unwrap())
        .captures("const ENVIRONMENT_SCALE = 2.0;");
    assert!(captures.is_some());
    let captures = captures.as_ref().unwrap();
    assert_eq!(&captures[1], "ENVIRONMENT_SCALE");
    assert_eq!(captures.get(2), None);
    assert_eq!(captures.get(3), None);
    assert_eq!(&captures[4], "2.0");
}

const F16_U16_MAX: f16 = f16::from_f32_const(u16::MAX as f32);

fn set_constants(shader_src: &str, constants: &[(&str, Cow<str>)]) -> String {
    let mut new_shader_src = String::new();
    for line in shader_src.lines() {
        let captures = CONST_RE
            .get_or_init(|| regex::Regex::new(RE_CONSTANTS).unwrap())
            .captures(line);
        if let Some(captures) = captures {
            let const_name = &captures[1];
            let ty = captures.get(3).map(|ty| ty.as_str());
            let new_value = constants.iter()
                .find(|(name, _)| *name == const_name)
                .map(|(_, value)| value);
            if let Some(new_value) = new_value {
                let new_line = if let Some(ty) = ty {
                    format!("const {const_name}: {ty} = {new_value};")
                }else{
                    format!("const {const_name} = {new_value};")
                };

                new_shader_src += &new_line;
            }else{
                new_shader_src += line;
            }
        }else{
            new_shader_src += line;
        }
        new_shader_src += "\n";
    }

    new_shader_src
}

fn set_texture_format(shader_src: &str, texture_formats: &[(&str, wgpu::TextureFormat)]) -> String {
    let mut new_shader_src = String::new();

    for line in shader_src.lines() {
        let captures = TEXTURE_STORAGE_RE
            .get_or_init(|| regex::Regex::new(RE_TEXTURE_STORAGE).unwrap())
            .captures(line);
        if let Some(captures) = captures {
            let texture_name = &captures[1];
            let new_ty = texture_formats.iter()
                .find(|(name, _)| *name == texture_name)
                .map(|(_, format)| format.to_wgsl_storage_str());
            if let Some(new_ty) = new_ty {
                let tex_storage = &captures[2];
                let rw = &captures[4];
                let new_line = format!("var {texture_name}: texture_storage_{tex_storage}<{new_ty}, {rw}>;");
                new_shader_src += &new_line;
            }else{
                new_shader_src += line;
            }
        }else{
            let captures = TEXTURE_CUBE_RE
                .get_or_init(|| regex::Regex::new(RE_TEXTURE_CUBE).unwrap())
                .captures(line);

            if let Some(captures) = captures {
                let texture_name = &captures[1];
                let new_ty = texture_formats.iter()
                    .find(|(name, _)| *name == texture_name)
                    .map(|(_, format)| format.to_wgsl_texture_str());
                if let Some(new_ty) = new_ty {
                    let ty = &captures[2];
                    let new_line = format!("var {texture_name}: texture_cube<{new_ty}>;");
                    new_shader_src += &new_line;
                }else{
                    new_shader_src += line;
                }
            }else{
                new_shader_src += line;
            }
        }
        new_shader_src += "\n";
    }

    new_shader_src
}

struct BakeParameters {
    num_samples: u16,
    strength: f32,
    contrast_correction: f32,
    brightness_correction: f32,
    saturation_correction: f32,
    hue_correction: f32,
}

impl BakeParameters {
    fn to_name_value(&self) -> [(&str, Cow<str>); 6] {
        [
            ("NUM_SAMPLES", Cow::Owned(format!("{}u", self.num_samples))),
            ("STRENGTH", Cow::Owned(format!("{:?}", self.strength))),
            ("CONTRAST_CORRECTION", Cow::Owned(format!("{:?}", self.contrast_correction))),
            ("BRIGHTNESS_CORRECTION", Cow::Owned(format!("{:?}", self.brightness_correction))),
            ("SATURATION_CORRECTION", Cow::Owned(format!("{:?}", self.saturation_correction))),
            ("HUE_CORRECTION", Cow::Owned(format!("{:?}", self.hue_correction))),
        ]
    }
}

// Runs a compute shader that converts an equirectangular input image into a cubemap
async fn equirectangular_to_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &DynamicImage,
    cubemap_side: u32,
    pixel_format: wgpu::TextureFormat,
) -> Option<wgpu::Texture> {
    // TODO: check if input is different
    let env_map_format = wgpu::TextureFormat::Rgba32Float;

    static EQUI_TO_CUBEMAP_SRC: &str = include_str!("shaders/equirectangular_to_cubemap.wgsl");
    let equi_to_cubemap_src = set_texture_format(EQUI_TO_CUBEMAP_SRC, &[
        ("equirectangular", env_map_format),
        ("cubemap_faces", pixel_format),
    ]);

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&equi_to_cubemap_src)),
    });

    let env_map = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some("Envmap"),
            size: wgpu::Extent3d{ width: env_map.width(), height: env_map.height(), depth_or_array_layers: 1},
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: env_map_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[]
        },
        env_map.as_bytes()
    );

    let env_map_view = env_map.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..wgpu::TextureViewDescriptor::default()
    });

    let cubemap = device.create_texture(
        &TextureDescriptor {
            label: Some("Cubemap"),
            size: wgpu::Extent3d{ width: cubemap_side, height: cubemap_side, depth_or_array_layers: 6},
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pixel_format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[]
        },
    );

    let cubemap_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(6),
        ..wgpu::TextureViewDescriptor::default()
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: pixel_format,
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
        ],
    });

    // A pipeline specifies the operation of a shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Equirectangular To Cubemap Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });


    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "equirectangular_to_cubemap",
    });


    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Equirectangular to Cubemap BindGroup"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&env_map_view),
        },wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&cubemap_view),
        }],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("Compute equirectangular to cubemap");
        cpass.dispatch_workgroups(cubemap_side, cubemap_side, 6); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::Wait);

    Some(cubemap)
}

fn generate_mipmaps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> wgpu::Texture {
    static GENERATE_MIPMAPS_SRC: &str = include_str!("shaders/generate_mipmaps.wgsl");
    let generate_mipmaps_src = set_texture_format(GENERATE_MIPMAPS_SRC, &[
        ("input", texture.format()),
        ("output", texture.format())
    ]);
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&generate_mipmaps_src)),
    });

    let max_side = texture.width().max(texture.height());
    let mip_level_count = (max_side as f32).log2().floor() as u32 + 1;
    let output = device.create_texture(
        &TextureDescriptor {
            label: Some("GenerateMipmapsOutput"),
            size: texture.size(),
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture.format(),
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[]
        },
    );


    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: texture.format(),
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: texture.format(),
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
        ],
    });



    // A pipeline specifies the operation of a shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("generate mipmaps Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });


    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "generate_mipmaps",
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_texture_to_texture(
        ImageCopyTexture{texture, mip_level: 0, origin: Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
        ImageCopyTexture{texture: &output, mip_level: 0, origin: Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
        wgpu::Extent3d { width: texture.width(), height: texture.height(), depth_or_array_layers: texture.depth_or_array_layers() }
    );

    let mut side = max_side >> 1;
    let mut level = 0;
    while side > 0 {
        let output_view = output.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(6),
            base_mip_level: level + 1,
            mip_level_count: Some(1),
            ..wgpu::TextureViewDescriptor::default()
        });
        let input_view = output.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(6),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..wgpu::TextureViewDescriptor::default()
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Generate mipmaps BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&output_view),
            }],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("Generate mipmaps");
            cpass.dispatch_workgroups(side, side, 6); // Number of cells to run, the (x,y,z) size of item being processed
        }

        side >>= 1;
        level += 1;
    }

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::Wait);

    output
}

// Downloads the data of a cubemap in GPU memory to a Vec<f32>. It returns the data
// and the number of levels that it downloaded
async fn download_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cubemap: &wgpu::Texture,
) -> Option<Vec<u8>>
{
    let mut result = vec![];
    let bytes_per_pixel = cubemap.format()
        .block_size(Some(wgpu::TextureAspect::All))
        .unwrap();

    // Will copy data from texture on GPU to staging buffer on CPU.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cubemap.width() as u64 * cubemap.height() as u64 * 6 * bytes_per_pixel as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let aux_texture = if cubemap.mip_level_count() > 1 {
        Some(device.create_texture(&TextureDescriptor {
            label: Some("Aux padded texture"),
            size: wgpu::Extent3d{
                width: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / bytes_per_pixel,
                height: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / bytes_per_pixel,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: cubemap.format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[]
        }))
    }else{
        None
    };

    for level in 0..cubemap.mip_level_count() {
        println!("Downloading level {level}");
        let level_side = cubemap.width() >> level;
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let (cubemap, level) = if level_side * bytes_per_pixel < wgpu::COPY_BYTES_PER_ROW_ALIGNMENT {
            println!("Copying from side: {level_side} to {}", aux_texture.as_ref().unwrap().width());
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTextureBase {
                    texture: cubemap,
                    mip_level: level,
                    origin: Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All
                },
                wgpu::ImageCopyTextureBase {
                    texture: aux_texture.as_ref().unwrap(),
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All
                },
                wgpu::Extent3d { width: level_side, height: level_side, depth_or_array_layers: 6 }
            );
            (aux_texture.as_ref().unwrap(), 0)
        }else{
            (cubemap, level)
        };

        let bytes_per_row = (level_side * bytes_per_pixel).max(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTextureBase {
                texture: cubemap,
                mip_level: level,
                origin: Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            wgpu::ImageCopyBufferBase{
                buffer: &staging_buffer,
                layout: ImageDataLayout{
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(level_side),
                }
            },
            wgpu::Extent3d { width: bytes_per_row / bytes_per_pixel, height: level_side, depth_or_array_layers: 6 }
        );

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));


        // Note that we're not calling `.await` here.
        // TODO: spawn and start next copy?
        let level_bytes = bytes_per_row as u64 * level_side as u64 * 6;
        let buffer_slice = staging_buffer.slice(..level_bytes);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = receiver.receive().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            if level_side * bytes_per_pixel < wgpu::COPY_BYTES_PER_ROW_ALIGNMENT {
                // We are using the auxiliary padded texture to download so we need to copy row by row
                for row in data
                    .chunks(aux_texture.as_ref().unwrap().width() as usize * bytes_per_pixel as usize)
                    .take(level_side as usize * 6)
                {
                    result.extend(&row[..level_side as usize * bytes_per_pixel as usize]);
                }
            }else{
                result.extend_from_slice(&data);
            }

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
        }else{
            return None
        }
    }

    // Returns data from buffer
    Some(result)
}

// Downloads the data of a texture in GPU memory to a Vec<f32>. It returns the data
// and the number of levels that it downloaded
async fn download_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Option<Vec<u8>>
{
    let mut result = vec![];
    let bytes_per_pixel = texture.format()
        .block_size(Some(wgpu::TextureAspect::All))
        .unwrap();

    // Will copy data from texture on GPU to staging buffer on CPU.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: texture.width() as u64 * texture.height() as u64 * bytes_per_pixel as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let bytes_per_row = texture.width() * bytes_per_pixel as u32;
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTextureBase {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        wgpu::ImageCopyBufferBase{
            buffer: &staging_buffer,
            layout: ImageDataLayout{
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(texture.height()),
            }
        },
        texture.size()
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));


    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        result.extend_from_slice(&data);

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
    }else{
        return None
    }

    // Returns data from buffer
    Some(result)
}

enum KtxVersion {
    _1,
    _2,
}

const GL_RGBA32F: u32 = 0x8814;
const GL_RGBA16F: u32 = 0x881A;
const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;

trait ToApi {
    fn to_gl(self) -> u32;
    fn to_vulkan(self) -> u32;
    fn to_wgsl_storage_str(self) -> &'static str ;
    fn to_wgsl_texture_str(self) -> &'static str;
}

impl ToApi for wgpu::TextureFormat {
    fn to_gl(self) -> u32 {
        match self {
            wgpu::TextureFormat::Rgba32Float => GL_RGBA32F,
            wgpu::TextureFormat::Rgba16Float => GL_RGBA16F,
            _ => todo!()
        }
    }

    fn to_vulkan(self) -> u32 {
        match self {
            wgpu::TextureFormat::Rgba32Float => VK_FORMAT_R32G32B32A32_SFLOAT,
            wgpu::TextureFormat::Rgba16Float => VK_FORMAT_R16G16B16A16_SFLOAT,
            _ => todo!()
        }
    }

    fn to_wgsl_storage_str(self) -> &'static str {
        match self {
            wgpu::TextureFormat::Rgba32Float => "rgba32float",
            wgpu::TextureFormat::Rgba16Float => "rgba16float",
            _ => todo!()
        }
    }

    fn to_wgsl_texture_str(self) -> &'static str {
        match self {
            wgpu::TextureFormat::Rgba32Float => "f32",
            wgpu::TextureFormat::Rgba16Float => "f32",
            _ => todo!()
        }
    }
}

fn write_cubemap_to_ktx(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str, ktx_version: KtxVersion) {
    let bytes_per_pixel = format
        .block_size(Some(wgpu::TextureAspect::All))
        .unwrap() as usize;

    let c_output_file = CString::new(output_file).unwrap();
    let mut create_info = libktx_rs_sys::ktxTextureCreateInfo {
        baseWidth: cubemap_side,
        baseHeight: cubemap_side,
        baseDepth: 1,
        numDimensions: 2,
        numLevels: cubemap_levels,
        numLayers: 1,
        numFaces: 6,
        generateMipmaps: false,
        glInternalformat: format.to_gl(),
        vkFormat: format.to_vulkan(),
        isArray: false,
        pDfd: ptr::null_mut(),
    };
    let texture: *mut ktxTexture;
    unsafe{
        match ktx_version {
            KtxVersion::_1 => {
                let mut texture_ktx1 = ptr::null_mut();
                ktxTexture1_Create(&mut create_info, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, &mut texture_ktx1);
                texture = texture_ktx1 as *mut ktxTexture;

            }
            KtxVersion::_2 => {
                let mut texture_ktx2 = ptr::null_mut();
                ktxTexture2_Create(&mut create_info, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, &mut texture_ktx2);
                texture = texture_ktx2 as *mut ktxTexture;
            }
        }

        let vtbl = &*(*texture).vtbl;
        let mut prev_end = 0;
        for level in 0..cubemap_levels {
            let level_side = (cubemap_side >> level) as usize;
            let face_size = level_side * level_side * bytes_per_pixel;
            for (face_idx, face) in cubemap_data[prev_end..]
                .chunks(level_side * level_side * bytes_per_pixel)
                .enumerate()
                .take(6)
            {
                (vtbl.SetImageFromMemory.unwrap())(texture, level, 0, face_idx as u32, face.as_ptr(), face_size);
            }
            prev_end += level_side * level_side * bytes_per_pixel * 6;
        }
        (vtbl.WriteToNamedFile.unwrap())(texture, c_output_file.as_ptr());
        (vtbl.Destroy.unwrap())(texture);
    }

    // Text ktx is saving correctly
    #[cfg(feature="test-ktx")]
    {
        use libktx_rs_sys::{ktxTexture_CreateFromNamedFile, ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, ktxTexture_GetData};
        let mut texture = ptr::null_mut();
        unsafe{
            let result = ktxTexture_CreateFromNamedFile(
                c_output_file.as_ptr(),
                ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                &mut texture
            );

            let layer = 0;
            let vtbl = &*(*texture).vtbl;
            for level in 0..cubemap_levels {
                let level_side = (cubemap_side >> level) as usize;
                for face_idx in 0..6 {
                    let mut offset  = 0;
                    let result = (vtbl.GetImageOffset.unwrap())(texture, level, layer, face_idx, &mut offset);
                    let face_data = ktxTexture_GetData(texture).offset(offset as isize);
                    let face_data = std::slice::from_raw_parts(face_data as *const u8, level_side * level_side * bytes_per_pixel);
                    let face = image_data_to_u16(face_data, format);
                    let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side as u32, level_side as u32, face).unwrap();
                    img.save(format!("{output_file}_face{}_level{}.png", face_idx, level)).unwrap();
                }
            }

            (vtbl.Destroy.unwrap())(texture);
        }
    }
}

// Writes the data of a cubemap as downloaded from GPU to a KTX2
fn write_cubemap_to_ktx1(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, format, cubemap_side, cubemap_levels, output_file, KtxVersion::_1)
}

// Writes the data of a cubemap as downloaded from GPU to a KTX2
fn write_cubemap_to_ktx2(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, format, cubemap_side, cubemap_levels, output_file, KtxVersion::_2)
}

// Writes the data of a cubemap as downloaded from GPU to a DDS
fn write_cubemap_to_dds(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) -> Result<(), dds::Error> {
    let bytes_per_pixel = format
        .block_size(Some(wgpu::TextureAspect::All))
        .unwrap() as usize;

    // Dds layout is transposed, each face with all it's levels but we have each level with
    // all it's faces so we need to transpose first
    let mut dds_data = vec![];
    for face_idx in 0..6 {
        let mut prev_end = 0;
        for level in 0..cubemap_levels {
            let level_side = (cubemap_side >> level) as usize;
            let face = cubemap_data[prev_end..]
                .chunks(level_side * level_side * bytes_per_pixel)
                .skip(face_idx)
                .next()
                .unwrap();
            dds_data.extend_from_slice(face);
            prev_end += level_side * level_side * bytes_per_pixel * 6;
        }
    }

    let ty = match format {
        wgpu::TextureFormat::Rgba32Float => dds::Type::Float,
        wgpu::TextureFormat::Rgba16Float => todo!(),
        _ => todo!()
    };

    let dds = dds::Builder::new(cubemap_side as usize, cubemap_side as usize, dds::Format::RGBA, ty)
        .is_cubemap_allfaces()
        .has_mipmaps(cubemap_levels as usize)
        .create(dds_data)?;
    dds.save(output_file)?;

    // Test dds is saving correctly
    #[cfg(feature="test-dds")]
    for face_idx in 0..6 {
        for level in 0..cubemap_levels {
            let level_side = cubemap_side >> level;
            let face = image_data_to_u16(dds.face(face_idx).unwrap().mipmap(level as usize).unwrap().data(), format);
            let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side, level_side, face).unwrap();
            img.save(format!("{output_file}_face{}_level{}.png", face_idx, level)).unwrap();
        }
    }

    Ok(())
}

// Bakes the IBL radiance map from an environment map. The input environment map and the output
// radiance map are cubemaps
async fn radiance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &wgpu::Texture,
    cubemap_side: u32,
    parameters: &BakeParameters,
) -> Option<wgpu::Texture> {
    static RADIANCE_SRC: &str = include_str!("shaders/ibl_bake.wgsl");
    let radiance_src = set_constants(RADIANCE_SRC, &parameters.to_name_value());
    let radiance_src = set_texture_format(&radiance_src, &[
        ("envmap", env_map.format()),
        ("output_faces", env_map.format())
    ]);
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&radiance_src)),
    });

    let env_map_view = env_map.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        mip_level_count: Some(env_map.mip_level_count()),
        ..wgpu::TextureViewDescriptor::default()
    });

    let max_mip = (cubemap_side as f32).log2().floor() as u32 + 1;
    let output = device.create_texture(
        &TextureDescriptor {
            label: Some("Radiance"),
            size: wgpu::Extent3d{ width: cubemap_side, height: cubemap_side, depth_or_array_layers: 6},
            mip_level_count: max_mip,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: env_map.format(),
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[]
        },
    );

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: env_map.format(),
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
        ],
    });
    let bind_group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None
            }
        ]
    });

    // A pipeline specifies the operation of a shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Radiance Layout"),
        bind_group_layouts: &[&bind_group_layout, &bind_group2_layout],
        push_constant_ranges: &[],
    });


    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "radiance",
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    #[repr(C)]
    #[derive(Pod, Copy, Clone, Zeroable)]
    struct RadianceData {
        mip_level: u32,
        max_mip: u32,
    }

    for mip_level in 0..max_mip {
        // TODO: doing all the cycles in one command encoder hangs the OS and outputs black
        let level_side = cubemap_side >> mip_level;
        println!("Processing radiance level {mip_level} side: {}", level_side);

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let output_level_view = output.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(6),
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..wgpu::TextureViewDescriptor::default()
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radiance BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&env_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_level_view),
                },
            ],
        });

        // TODO: Create the buffer once and rewrite
        let uniforms = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Buffer"),
            contents: bytemuck::cast_slice(&[RadianceData {
                mip_level,
                max_mip,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radiance uniforms bind group"),
            layout: &bind_group2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding()
                }
            ]
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Compute radiance level {}", mip_level)),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_bind_group(1, &bind_group_uniforms, &[]);
            cpass.insert_debug_marker(&format!("Compute radiance level {}", mip_level));
            cpass.dispatch_workgroups(level_side, level_side, 6); // Number of cells to run, the (x,y,z) size of item being processed
        }

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        // Poll the device in a blocking manner so that our future resolves.
        device.poll(wgpu::Maintain::Wait);
    }

    Some(output)
}

// Bakes the IBL irradiance map from an environment map. The input environment map and the output
// radiance map are cubemaps
async fn irradiance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &wgpu::Texture,
    cubemap_side: u32,
    parameters: &BakeParameters,
) -> Option<wgpu::Texture> {
    static IRRADIANCE_SRC: &str = include_str!("shaders/ibl_bake.wgsl");
    let irradiance_src = set_constants(&IRRADIANCE_SRC, &parameters.to_name_value());
    let irradiance_src = set_texture_format(&irradiance_src, &[
        ("envmap", env_map.format()),
        ("output_faces", env_map.format())
    ]);
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&irradiance_src)),
    });

    let env_map_view = env_map.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        mip_level_count: Some(env_map.mip_level_count()),
        ..wgpu::TextureViewDescriptor::default()
    });

    let output = device.create_texture(
        &TextureDescriptor {
            label: Some("Irradiance"),
            size: wgpu::Extent3d{ width: cubemap_side, height: cubemap_side, depth_or_array_layers: 6},
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: env_map.format(),
            usage: wgpu::TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[]
        },
    );

    let output_view = output.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        array_layer_count: Some(6),
        ..wgpu::TextureViewDescriptor::default()
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: env_map.format(),
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
        ],
    });

    // A pipeline specifies the operation of a shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Irradiance Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });


    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "irradiance",
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    println!("Processing irradiance");

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Irradiance BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&env_map_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute irradiance"),
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("Compute irradiance");
        cpass.dispatch_workgroups(cubemap_side, cubemap_side, 6); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::Wait);

    Some(output)
}

// Calculates the GGX LUT
async fn ggx_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    side: u32,
    num_samples: u16,
    format: wgpu::TextureFormat,
) -> Option<wgpu::Texture> {
    static LUT_SRC: &str = include_str!("shaders/ggx_lut.wgsl");
    let lut_src = set_constants(&LUT_SRC, &[("NUM_SAMPLES", Cow::Owned(num_samples.to_string() + "u"))]);
    let lut_src = set_texture_format(&lut_src, &[
        ("lut", format)
    ]);

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&lut_src)),
    });

    let output = device.create_texture(
        &TextureDescriptor {
            label: Some("LUT"),
            size: wgpu::Extent3d{ width: side, height: side, depth_or_array_layers: 1},
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[]
        },
    );

    let output_view = output.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..wgpu::TextureViewDescriptor::default()
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None,
            },
        ],
    });

    // A pipeline specifies the operation of a shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("LUT Layout"),
        bind_group_layouts: &[
            &bind_group_layout
        ],
        push_constant_ranges: &[],
    });


    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "compute_lut",
    });

    println!("Processing lut");

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Instantiates the bind group, once again specifying the binding of buffers.
    // let bind_group_layout = compute_pipeline.get_bind_group_layout(2);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("LUT BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute GGX LUT"),
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("Compute GGX LUT");
        cpass.dispatch_workgroups(side, side, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::Wait);

    Some(output)
}

fn image_data_to_u16(image_data: &[u8], format: wgpu::TextureFormat) -> Vec<u16> {
    match format {
        wgpu::TextureFormat::Rgba32Float => bytemuck::cast_slice::<_, f32>(&image_data).chunks(4)
            .flat_map(|c| [
                (c[0] * u16::MAX as f32) as u16,
                (c[1] * u16::MAX as f32) as u16,
                (c[2] * u16::MAX as f32) as u16
            ]).collect(),
        wgpu::TextureFormat::Rgba16Float => bytemuck::cast_slice::<_, f16>(&image_data).chunks(4)
            .flat_map(|c| [
                (c[0].to_f32() * u16::MAX as f32) as u16,
                (c[1].to_f32() * u16::MAX as f32) as u16,
                (c[2].to_f32() * u16::MAX as f32) as u16,
            ]).collect(),
        _ => unreachable!(),
    }
}

#[async_std::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = clap::Command::new("skylights")
        .arg(
            clap::Arg::new("input-image")
                .required_unless_present("lut")
                .help("Environment map to process")
        )
        .arg(
            clap::Arg::new("cubemap-side")
                .default_value("1024")
                .short('x')
                .help("Side of the output cubemaps in pixels")
        )
        .arg(
            clap::Arg::new("pixel-format")
                .default_value("rgba16f")
                .value_parser(["rgba32f", "rgba16f"])
                .short('f')
                .help("Output cubemaps pixel format")
        )
        .arg(
            clap::Arg::new("encoding")
                .short('e')
                .value_parser(["ktx1", "ktx2", "dds", "png"])
                .default_value("ktx2")
                .help("Output cubemaps image encoding")
        )
        .arg(
            clap::Arg::new("num-samples")
                .default_value("128")
                .short('n')
                .value_parser(clap::value_parser!(u16))
                .help("Number of samples per pixel when calculating radiance and irradiance maps")
        )
        .arg(
            clap::Arg::new("strength")
                .default_value("1")
                .short('m')
                .value_parser(clap::value_parser!(f32))
                .help("Scales the final baked value in the radiance and irradiance maps")
        )
        .arg(
            clap::Arg::new("contrast")
                .default_value("1")
                .short('c')
                .value_parser(clap::value_parser!(f32))
                .help("Corrects the contrast of the final color")
        )
        .arg(
            clap::Arg::new("brightness")
                .default_value("1")
                .short('b')
                .value_parser(clap::value_parser!(f32))
                .help("Corrects the brightness of the final color")
        )
        .arg(
            clap::Arg::new("saturation")
                .default_value("1")
                .short('s')
                .value_parser(clap::value_parser!(f32))
                .help("Corrects the saturation of the final color 0: grayscale and 1: original color")
        )
        .arg(
            clap::Arg::new("hue")
                .default_value("0")
                .value_parser(clap::value_parser!(f32))
                .short('u')
                .help("Corrects the hue of the final color in degrees. [possible values: 0..360]")
        )
        .arg(clap::arg!(--lut -l "Computes GGX LUT"))
        .get_matches();

    let input_image: Option<&String> = args.get_one("input-image");

    let cubemap_side = args.get_one::<String>("cubemap-side")
        .unwrap()
        .parse()
        .expect("cubemap-side must be a numeric value"); // TODO: can be enforced in clap?

    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    let adapter = instance.enumerate_adapters(wgpu::Backends::all())
        .inspect(|adapter| { dbg!(adapter.get_info()); })
        .find(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu);

    let adapter = if let Some(adapter) = adapter {
        adapter
    }else{
        println!("Couldn't find any DiscreteGpu trying to use default HighPerformance adapter");
        instance
            .request_adapter(&wgpu::RequestAdapterOptions{
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or_else(|| "Error requesting adapter")
            .map_err(|err| anyhow::anyhow!(err))
            .context("Requesting adapter")?
    };

    println!("Using adapter: {:?}", adapter.get_info());

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::PUSH_CONSTANTS,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    let pixel_format = match args.get_one::<String>("pixel-format").unwrap().as_str() {
        "rgba32f" => wgpu::TextureFormat::Rgba32Float,
        "rgba16f" => wgpu::TextureFormat::Rgba16Float,
        _ => unreachable!()
    };

    if *args.get_one("lut").unwrap() {
        let num_samples = *args.get_one("num-samples").unwrap();
        let lut = ggx_lut(&device, &queue, cubemap_side, num_samples, pixel_format).await.unwrap();

        let lut_data = download_texture(&device, &queue, &lut).await.unwrap();
        let data_u16 = image_data_to_u16(&lut_data, pixel_format);
        let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, data_u16).unwrap();
        img.save("ggx_lut.png").unwrap();

        if input_image.is_none() {
            return Ok(());
        }
    }

    let input_image: &String = input_image.unwrap();
    if !Path::new(input_image).exists().await {
        return Err(anyhow::anyhow!(format!("Input image file \"{}\" doesn't exist", input_image)));
    }

    let encoding = args.get_one::<String>("encoding").unwrap().as_str();
    let bake_parameters = BakeParameters {
        num_samples: *args.get_one("num-samples").unwrap(),
        strength: *args.get_one("strength").unwrap(),
        contrast_correction: *args.get_one("contrast").unwrap(),
        brightness_correction: *args.get_one("brightness").unwrap(),
        saturation_correction: *args.get_one("saturation").unwrap(),
        hue_correction: *args.get_one("hue").unwrap(),
    };


    // Load environment map
    let mut img_file = File::open(input_image).await?;
    let mut img_buffer = vec![];
    img_file.read_to_end(&mut img_buffer).await?;
    let env_map = spawn_blocking(move || {
        // image::load_from_memory(&img_buffer)
        libhdr::Hdr::from_bytes(&img_buffer)
    }).await?;
    let rgba_env_map = env_map.rgb()
        .chunks(3)
        .flat_map(|c| [c[0], c[1], c[2], 1.])
        .collect();
    let env_map = DynamicImage::ImageRgba32F(
        ImageBuffer::from_vec(env_map.width(), env_map.height(), rgba_env_map)
            .ok_or_else(|| anyhow::anyhow!("Couldn't transform hdr into dynamic image"))?
    );


    let bytes_per_pixel = pixel_format
        .block_size(Some(wgpu::TextureAspect::All))
        .unwrap() as usize;

    // Convert equirectangular to cubemap
    let env_map = equirectangular_to_cubemap(&device, &queue, &env_map, cubemap_side, pixel_format).await.unwrap();
    // generate mipmaps for the environment map
    let env_map = generate_mipmaps(&device, &queue, &env_map);

    // Download environment map data
    let env_map_data = download_cubemap(&device, &queue, &env_map).await.unwrap();
    match encoding {
        "png" => {
            // Save as individual images per face
            for (idx, face) in env_map_data
                .chunks(cubemap_side as usize * cubemap_side as usize * bytes_per_pixel)
                .enumerate()
            {
                let face_u16 = image_data_to_u16(&face, pixel_format);
                let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, face_u16).unwrap();
                img.save(format!("face{}.png", idx)).unwrap();
            }
        }

        "dds" => write_cubemap_to_dds(&env_map_data, pixel_format, cubemap_side, env_map.mip_level_count(), "skybox.dds")?,

        "ktx1" => write_cubemap_to_ktx1(&env_map_data, pixel_format, cubemap_side, env_map.mip_level_count(), "skybox.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&env_map_data, pixel_format, cubemap_side, env_map.mip_level_count(), "skybox.ktx"),

        _ => unreachable!()
    }


    // Calculate radiance
    let radiance = radiance(&device, &queue, &env_map, cubemap_side, &bake_parameters).await.unwrap();

    // Download radiance data
    let radiance_data = download_cubemap(&device, &queue, &radiance).await.unwrap();
    match encoding {
        "png" => {
            // Save as individual images per face
            let mut prev_end = 0;
            for level in 0..radiance.mip_level_count() {
                let level_side = (cubemap_side >> level) as usize;
                for (idx, face) in radiance_data[prev_end..]
                    .chunks(level_side * level_side * bytes_per_pixel)
                    .enumerate()
                    .take(6)
                {
                    let face_u16 = image_data_to_u16(&face, pixel_format);
                    let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side as u32, level_side as u32, face_u16).unwrap();
                    img.save(format!("radiance_face{}_level{}.png", idx, level)).unwrap();
                }
                prev_end += level_side * level_side * bytes_per_pixel * 6;
            }
        }

        "dds" => write_cubemap_to_dds(&radiance_data, pixel_format, cubemap_side, radiance.mip_level_count(), "radiance.dds")?,

        "ktx1" => write_cubemap_to_ktx1(&radiance_data, pixel_format, cubemap_side, radiance.mip_level_count(), "radiance.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&radiance_data, pixel_format, cubemap_side, radiance.mip_level_count(), "radiance.ktx"),

        _ => unreachable!()
    }



    // Calculate irradiance
    let irradiance = irradiance(&device, &queue, &env_map, cubemap_side, &bake_parameters).await.unwrap();

    // Download irradiance data
    let irradiance_data = download_cubemap(&device, &queue, &irradiance).await.unwrap();
    match encoding {
        "png" => {
            // Save as individual images per face
            for (idx, face) in irradiance_data
                .chunks(cubemap_side as usize * cubemap_side as usize * bytes_per_pixel)
                .enumerate()
                .take(6)
            {
                let face_u16 = image_data_to_u16(&face, pixel_format);
                let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, face_u16).unwrap();
                img.save(format!("irradiance_face{}.png", idx)).unwrap();
            }
        }

        "dds" => write_cubemap_to_dds(&irradiance_data, pixel_format, cubemap_side, 1, "irradiance.dds")?,

        "ktx1" => write_cubemap_to_ktx1(&irradiance_data, pixel_format, cubemap_side, 1, "irradiance.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&irradiance_data, pixel_format, cubemap_side, 1, "irradiance.ktx"),

        _ => unreachable!()
    }

    Ok(())
}

