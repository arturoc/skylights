use std::{borrow::Cow, error::Error, mem::size_of};
use image::{DynamicImage, ImageBuffer, Rgb};
use wgpu::{util::DeviceExt, TextureDescriptor, TextureFormat, TextureUsages, Origin3d, ImageDataLayout};
use async_std::{prelude::*, fs::File, task::spawn_blocking, path::Path};

async fn equirectangular_to_cubemap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env_map: &DynamicImage,
    cubemap_side: u32,
) -> Option<Vec<f32>> {
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

    let cube_map_size = wgpu::Extent3d{ width: cubemap_side, height: cubemap_side, depth_or_array_layers: 6};
    let cube_map = device.create_texture(
        &TextureDescriptor {
            label: Some("Cubemap"),
            size: cube_map_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[wgpu::TextureFormat::Rgba32Float]
        },
    );

    let cube_map_view = cube_map.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..wgpu::TextureViewDescriptor::default()
    });

    let size = cubemap_side as u64 * cubemap_side as u64 * 6 * 4 * size_of::<f32>() as u64;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
                    view_dimension: wgpu::TextureViewDimension::D3
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Equirectangular To Cubemap Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
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
            resource: wgpu::BindingResource::TextureView(&cube_map_view),
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
        cpass.insert_debug_marker("Compute Equirectangular to Cubemap");
        cpass.dispatch_workgroups(cubemap_side, cubemap_side, 6); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTextureBase {
            texture: &cube_map,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        wgpu::ImageCopyBufferBase{
            buffer: &staging_buffer,
            layout: ImageDataLayout{
                offset: 0,
                bytes_per_row: Some(cubemap_side * 4 * size_of::<f32>() as u32),
                rows_per_image: Some(cubemap_side),
            }
        },
        cube_map_size
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
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
        .get_matches();

    let input_image: &String = args.get_one("input-image").unwrap();
    if !Path::new(input_image).exists().await {
        Err(format!("Input image file \"{}\" doesn't exist", input_image))?
    }

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

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or_else(|| "Error requesting adapter")?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    dbg!(device.limits().max_texture_dimension_2d);

    let cubemap_side = 1024;
    let cubemap = equirectangular_to_cubemap(&device, &queue, &env_map, cubemap_side).await.unwrap();
    for (idx, face) in cubemap.chunks(cubemap_side as usize*cubemap_side as usize*4).enumerate() {
        // let face0 = cubemap[0 .. cubemap_side as usize*cubemap_side as usize*4]
        let face = face.chunks(4)
            .flat_map(|c| [
                (c[0] * u16::MAX as f32) as u16,
                (c[1] * u16::MAX as f32) as u16,
                (c[2] * u16::MAX as f32) as u16
            ]).collect();
        let img: ImageBuffer<Rgb<u16>, _> = ImageBuffer::from_vec(cubemap_side, cubemap_side, face).unwrap();
        img.save(format!("face{}.png", idx)).unwrap();
    }

    Ok(())
}

