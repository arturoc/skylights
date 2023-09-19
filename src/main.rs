use std::{borrow::Cow, error::Error, mem::size_of, ptr, ffi::CString};
use bytemuck::{Pod, Zeroable};
use image::{DynamicImage, ImageBuffer, Rgb};
use libktx_rs_sys::{ktxTexture2_Create, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE};
use wgpu::{util::{DeviceExt, BufferInitDescriptor}, TextureDescriptor, TextureFormat, TextureUsages, Origin3d, ImageDataLayout, ImageCopyTexture};
use async_std::{prelude::*, fs::File, task::spawn_blocking, path::Path};

// Runs a compute shader that converts an equirectangular input image into a cubemap
async fn equirectangular_to_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &DynamicImage,
    cubemap_side: u32,
) -> Option<wgpu::Texture> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/equirectangular_to_cubemap.wgsl"))),
    });

    let env_map = device.create_texture_with_data(
        queue,
        &TextureDescriptor {
            label: Some("Envmap"),
            size: wgpu::Extent3d{ width: env_map.width(), height: env_map.height(), depth_or_array_layers: 1},
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba32Float]
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
            format: wgpu::TextureFormat::Rgba32Float,
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
                    format: TextureFormat::Rgba32Float,
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
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/generate_mipmaps.wgsl"))),
    });

    let max_side = texture.width().max(texture.height());
    let mip_level_count = (max_side as f32).log2().floor() as u32 + 1;
    let output = device.create_texture(
        &TextureDescriptor {
            label: Some("GenerateMipmapsOutput"),
            size: wgpu::Extent3d{ width: texture.width(), height: texture.height(), depth_or_array_layers: texture.depth_or_array_layers()},
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
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
                    format: TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2Array
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
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
) -> Option<Vec<f32>>
{


    let mut result = vec![];

    // Will copy data from texture on GPU to staging buffer on CPU.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: cubemap.width() as u64 * cubemap.height() as u64 * 6 * 4 * size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let aux_texture = if cubemap.mip_level_count() > 1 {
        Some(device.create_texture(&TextureDescriptor {
            label: Some("Aux padded texture"),
            size: wgpu::Extent3d{
                width: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / (4 * size_of::<f32>()) as u32,
                height: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / (4 * size_of::<f32>()) as u32,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
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
        let (cubemap, level) = if (level_side * 4 * size_of::<f32>() as u32) < wgpu::COPY_BYTES_PER_ROW_ALIGNMENT {
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

        let bytes_per_row = (level_side * 4 * size_of::<f32>() as u32).max(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
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
            wgpu::Extent3d { width: bytes_per_row / (4 * size_of::<f32>() as u32), height: level_side, depth_or_array_layers: 6 }
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
            if (level_side * 4 * size_of::<f32>() as u32) < wgpu::COPY_BYTES_PER_ROW_ALIGNMENT {
                // We are using the auxiliary padded texture to download so we need to copy row by row
                for row in bytemuck::cast_slice::<_, f32>(&data)
                    .chunks(aux_texture.as_ref().unwrap().width() as usize * 4)
                    .take(level_side as usize * 6)
                {
                    result.extend(&row[..level_side as usize * 4]);
                }
            }else{
                result.extend_from_slice(bytemuck::cast_slice(&data));
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

enum KtxVersion {
    _1,
    _2,
}

fn write_cubemap_to_ktx(cubemap_data: &[f32], cubemap_side: u32, cubemap_levels: u32, output_file: &str, ktx_version: KtxVersion) {
    let c_output_file = CString::new(output_file).unwrap();
    const GL_RGBA32F: u32 = 0x8814;
    const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;
    let mut create_info = libktx_rs_sys::ktxTextureCreateInfo {
        baseWidth: cubemap_side,
        baseHeight: cubemap_side,
        baseDepth: 1,
        numDimensions: 2,
        numLevels: cubemap_levels,
        numLayers: 1,
        numFaces: 6,
        generateMipmaps: false,
        glInternalformat: GL_RGBA32F,
        vkFormat: VK_FORMAT_R32G32B32A32_SFLOAT,
        isArray: false,
        pDfd: ptr::null_mut(),
    };
    let texture: *mut ktxTexture;
    unsafe{
        match ktx_version {
            KtxVersion::_1 => {
                let mut texture_ktx1 = ptr::null_mut();
                ktxTexture1_Create(&mut create_info, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, &mut texture_ktx1);
                texture = texture_ktx1 as *mut _;

            }
            KtxVersion::_2 => {
                let mut texture_ktx2 = ptr::null_mut();
                ktxTexture2_Create(&mut create_info, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, &mut texture_ktx2);
                texture = texture_ktx2 as *mut _;
            }
        }

        let vtbl = &*(*texture).vtbl;
        let mut prev_end = 0;
        for level in 0..cubemap_levels {
            let level_side = cubemap_side >> level;
            let face_size = level_side as usize * level_side as usize * 4 * size_of::<f32>();
            for (face_idx, face) in cubemap_data[prev_end..]
                .chunks(level_side as usize * level_side as usize * 4)
                .enumerate()
                .take(6)
            {
                (vtbl.SetImageFromMemory.unwrap())(texture, level, 0, face_idx as u32, face.as_ptr() as *const u8, face_size);
            }
            prev_end += level_side as usize * level_side as usize * 4 * 6;
        }
        (vtbl.WriteToNamedFile.unwrap())(texture, c_output_file.as_ptr());
        (vtbl.Destroy.unwrap())(texture);
    }

    // Text ktx is saving correctly
    // use libktx_rs_sys::{ktxTexture_CreateFromNamedFile, ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, ktxTexture_GetData};
    // let mut texture = ptr::null_mut();
    // unsafe{
    //     let result = ktxTexture_CreateFromNamedFile(
    //         c_output_file.as_ptr(),
    //         ktxTextureCreateFlagBits_KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
    //         &mut texture
    //     );

    //     let layer = 0;
    //     let vtbl = &*(*texture).vtbl;
    //     for level in 0..cubemap_levels {
    //         let level_side = (cubemap_side >> level) as usize;
    //         for face_idx in 0..6 {
    //             let mut offset  = 0;
    //             let result = (vtbl.GetImageOffset.unwrap())(texture, level, layer, face_idx, &mut offset);
    //             let face_data = ktxTexture_GetData(texture).offset(offset as isize);
    //             let face_data = std::slice::from_raw_parts(face_data as *const f32, level_side * level_side * 4);
    //             let face = face_data.chunks(4)
    //                 .flat_map(|c| [
    //                     (c[0] * u16::MAX as f32) as u16,
    //                     (c[1] * u16::MAX as f32) as u16,
    //                     (c[2] * u16::MAX as f32) as u16
    //                 ]).collect::<Vec<_>>();
    //             dbg!(level_side, face.len() / 3, face_data.len() / 4, level_side * level_side);
    //             let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side as u32, level_side as u32, face).unwrap();
    //             img.save(format!("{output_file}_face{}_level{}.png", face_idx, level)).unwrap();
    //         }
    //     }

    //     (vtbl.Destroy.unwrap())(texture);
    // }
}

// Writes the data of a cubemap as downloaded from GPU to a KTX2
fn write_cubemap_to_ktx1(cubemap_data: &[f32], cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, cubemap_side, cubemap_levels, output_file, KtxVersion::_1)
}

// Writes the data of a cubemap as downloaded from GPU to a KTX2
fn write_cubemap_to_ktx2(cubemap_data: &[f32], cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, cubemap_side, cubemap_levels, output_file, KtxVersion::_2)
}

// Writes the data of a cubemap as downloaded from GPU to a DDS
fn write_cubemap_to_dds(cubemap_data: &[f32], cubemap_side: u32, cubemap_levels: u32, output_file: &str) -> Result<(), dds::Error> {
    // Dds layout is transposed, each face with all it's levels but we have each level with
    // all it's faces so we need to transpose first
    let mut dds_data = vec![];
    for face_idx in 0..6 {
        let mut prev_end = 0;
        for level in 0..cubemap_levels {
            let level_side = cubemap_side >> level;
            if cubemap_data.len() <= prev_end {
                break;
            }
            let face = cubemap_data[prev_end..]
                .chunks(level_side as usize * level_side as usize * 4).skip(face_idx)
                .next()
                .unwrap();
            dds_data.extend_from_slice(face);
            prev_end += level_side as usize * level_side as usize * 4 * 6;
        }
    }

    // Save as cubemap dds
    let radiance_datau8 = unsafe{
        std::slice::from_raw_parts(dds_data.as_ptr() as *const u8, dds_data.len() * size_of::<f32>())
    };
    let dds = dds::Builder::new(cubemap_side as usize, cubemap_side as usize, dds::Format::RGBA, dds::Type::Float)
        .is_cubemap_allfaces()
        .has_mipmaps(cubemap_levels as usize)
        .create(radiance_datau8)?;
    dds.save(output_file)



    // Test dds is saving correctly
    // for face_idx in 0..6 {
    //     for level in 0..read_levels {
    //         let level_side = cubemap_side >> level;
    //         let face = dds.face(face_idx).unwrap().mipmap(level as usize).unwrap().data().chunks(4 * size_of::<f32>())
    //             .flat_map(|c| {
    //                 let c = unsafe{ std::slice::from_raw_parts(c.as_ptr() as *const f32, 4) };
    //                 [
    //                     (c[0] * u16::MAX as f32) as u16,
    //                     (c[1] * u16::MAX as f32) as u16,
    //                     (c[2] * u16::MAX as f32) as u16
    //                 ]
    //             }).collect();
    //         let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side, level_side, face).unwrap();
    //         img.save(format!("radiance_face{}_level{}.png", face_idx, level)).unwrap();
    //     }
    // }
}

// Bakes the IBL radiance map from an environment map. The input environment map and the output
// radiance map are cubemaps
async fn radiance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &wgpu::Texture,
    cubemap_side: u32,
) -> Option<wgpu::Texture> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/ibl_bake.wgsl"))),
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
            format: wgpu::TextureFormat::Rgba32Float,
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
                    format: TextureFormat::Rgba32Float,
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
        label: Some("Equirectangular To Cubemap Layout"),
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
) -> Option<wgpu::Texture> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/ibl_bake.wgsl"))),
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
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
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
                    format: TextureFormat::Rgba32Float,
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

    #[repr(C)]
    #[derive(Pod, Copy, Clone, Zeroable)]
    struct RadianceData {
        mip_level: u32,
        max_mip: u32,
    }

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

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let args = clap::Command::new("skylights")
        .arg(
            clap::Arg::new("input-image")
                .required(true)
                .help("Environment map to process")
        )
        .arg(
            clap::Arg::new("cubemap-side")
                .default_value("1024")
                .short('s')
                .help("Side of the output cubemaps in pixels")
        )
        .arg(
            clap::Arg::new("output-format")
                .default_value("ktx")
                .short('f')
                // .value_names(["dds", "ktx"])
                .value_parser(["ktx1", "ktx2", "dds", "png"])
                .default_value("ktx2")
                .help("Output cubemaps format")
        )
        .get_matches();

    let input_image: &String = args.get_one("input-image").unwrap();
    if !Path::new(input_image).exists().await {
        Err(format!("Input image file \"{}\" doesn't exist", input_image))?
    }

    let cubemap_side = args.get_one::<String>("cubemap-side")
        .unwrap()
        .parse()
        .expect("cubemap-side must be a numeric value"); // TODO: can be enforced in clap?
    let output_format = args.get_one::<String>("output-format").unwrap().as_str();


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
            .ok_or_else(|| "Couldn't transform hdr into dynamic image")?
    );

    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    let adapter = instance.enumerate_adapters(wgpu::Backends::all())
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
            .ok_or_else(|| "Error requesting adapter")?
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

    // Convert equirectangular to cubemap
    let env_map = equirectangular_to_cubemap(&device, &queue, &env_map, cubemap_side).await.unwrap();

    // Download environment map data
    let env_map_data = download_cubemap(&device, &queue, &env_map).await.unwrap();
    match output_format {
        "png" => {
            // Save as individual images per face
            for (idx, face) in env_map_data
                .chunks(cubemap_side as usize*cubemap_side as usize*4)
                .enumerate()
            {
                let face = face.chunks(4)
                    .flat_map(|c| [
                        (c[0] * u16::MAX as f32) as u16,
                        (c[1] * u16::MAX as f32) as u16,
                        (c[2] * u16::MAX as f32) as u16
                    ]).collect();
                let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, face).unwrap();
                img.save(format!("face{}.png", idx)).unwrap();
            }
        }

        "dds" => {
            // Save as cubemap dds
            let cubemap_datau8 = unsafe{
                std::slice::from_raw_parts(env_map_data.as_ptr() as *const u8, env_map_data.len() * size_of::<f32>())
            };
            dds::Builder::new(cubemap_side as usize, cubemap_side as usize, dds::Format::RGBA, dds::Type::Float)
                .is_cubemap_allfaces()
                .create(cubemap_datau8)?
                .save("skybox.dds")?;
        }

        "ktx1" => write_cubemap_to_ktx1(&env_map_data, cubemap_side, 1, "skybox.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&env_map_data, cubemap_side, 1, "skybox.ktx"),

        _ => unreachable!()
    }

    // generate mipmaps for the environment map
    let env_map = generate_mipmaps(&device, &queue, &env_map);

    // Calculate radiance
    let radiance = radiance(&device, &queue, &env_map, cubemap_side).await.unwrap();

    // Download radiance data
    let radiance_data = download_cubemap(&device, &queue, &radiance).await.unwrap();
    match output_format {
        "png" => {
            // Save as individual images per face
            let mut prev_end = 0;
            for level in 0..radiance.mip_level_count() {
                let level_side = cubemap_side >> level;
                for (idx, face) in radiance_data[prev_end..]
                    .chunks(level_side as usize * level_side as usize * 4)
                    .enumerate()
                    .take(6)
                {
                    let face: Vec<_> = face.chunks(4)
                        .flat_map(|c| [
                            (c[0] * u16::MAX as f32) as u16,
                            (c[1] * u16::MAX as f32) as u16,
                            (c[2] * u16::MAX as f32) as u16
                        ]).collect();
                    let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(level_side, level_side, face).unwrap();
                    img.save(format!("radiance_face{}_level{}.png", idx, level)).unwrap();
                }
                prev_end += level_side as usize * level_side as usize * 4 * 6;
            }
        }

        "dds" => write_cubemap_to_dds(&radiance_data, cubemap_side, radiance.mip_level_count(), "radiance.dds")?,

        "ktx1" => write_cubemap_to_ktx1(&radiance_data, cubemap_side, radiance.mip_level_count(), "radiance.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&radiance_data, cubemap_side, radiance.mip_level_count(), "radiance.ktx"),

        _ => unreachable!()
    }



    // Calculate irradiance
    let irradiance = irradiance(&device, &queue, &env_map, cubemap_side).await.unwrap();

    // Download irradiance data
    let irradiance_data = download_cubemap(&device, &queue, &irradiance).await.unwrap();
    match output_format {
        "png" => {
            // Save as individual images per face
            for (idx, face) in irradiance_data
                .chunks(cubemap_side as usize * cubemap_side as usize * 4)
                .enumerate()
                .take(6)
            {
                let face = face.chunks(4)
                    .flat_map(|c| [
                        (c[0] * u16::MAX as f32) as u16,
                        (c[1] * u16::MAX as f32) as u16,
                        (c[2] * u16::MAX as f32) as u16
                    ]).collect();
                let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, face).unwrap();
                img.save(format!("irradiance_face{}.png", idx)).unwrap();
            }
        }

        "dds" => write_cubemap_to_dds(&irradiance_data, cubemap_side, 1, "irradiance.dds")?,

        "ktx1" => write_cubemap_to_ktx1(&irradiance_data, cubemap_side, 1, "irradiance.ktx"),

        "ktx2" => write_cubemap_to_ktx2(&irradiance_data, cubemap_side, 1, "irradiance.ktx"),

        _ => unreachable!()
    }

    Ok(())
}

