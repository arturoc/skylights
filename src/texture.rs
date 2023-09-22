use std::{ptr, ffi::CString};
use libktx_rs_sys::{ktxTexture2_Create, ktxTextureCreateStorageEnum_KTX_TEXTURE_CREATE_ALLOC_STORAGE, ktxTexture1_Create, ktxTexture};
use anyhow::Result;

const GL_RGBA32F: u32 = 0x8814;
const GL_RGBA16F: u32 = 0x881A;
const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;

pub trait ToApi {
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


enum KtxVersion {
    _1,
    _2,
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
pub fn write_cubemap_to_ktx1(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, format, cubemap_side, cubemap_levels, output_file, KtxVersion::_1)
}

// Writes the data of a cubemap as downloaded from GPU to a KTX2
pub fn write_cubemap_to_ktx2(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) {
    write_cubemap_to_ktx(cubemap_data, format, cubemap_side, cubemap_levels, output_file, KtxVersion::_2)
}

// Writes the data of a cubemap as downloaded from GPU to a DDS
pub fn write_cubemap_to_dds(cubemap_data: &[u8], format: wgpu::TextureFormat, cubemap_side: u32, cubemap_levels: u32, output_file: &str) -> Result<(), dds::Error> {
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
        wgpu::TextureFormat::Rgba32Float => dds::Type::Float32,
        wgpu::TextureFormat::Rgba16Float => dds::Type::Float16,
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
