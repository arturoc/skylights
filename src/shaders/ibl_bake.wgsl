@group(0)
@binding(0)
var envmap: texture_cube<f32>;

@group(0)
@binding(1)
var envmap_sampler: sampler;

@group(0)
@binding(2)
var output_faces: texture_storage_2d_array<rgba32float, write>;

struct RadianceData {
	mip_level: u32,
	max_mips: u32,
}
@group(1)
@binding(0)
var<uniform> radiance_data: RadianceData;

const INV_ATAN: vec2<f32> = vec2(0.1591, 0.3183);
const CP_UDIR = 0u;
const CP_VDIR = 1u;
const CP_FACEAXIS = 2u;
const M_PI = 3.1415926535897932384626433832795;
const M_INV_PI = 0.31830988618;
const MAX_SAMPLES = 1024u;

fn face_2d_mapping(face: u32) -> array<vec3f, 3> {
    //XPOS face
	if(face==0u) {
		return array<vec3f, 3>(
		     vec3(0.,  0., -1.),   //u towards negative Z
		     vec3(0., -1.,  0.),   //v towards negative Y
		     vec3(1.,  0.,  0.)
        );  //pos X axis
    }
    //XNEG face
	if(face==1u) {
		return array<vec3f, 3>(
		      vec3(0.,  0.,  1.),   //u towards positive Z
		      vec3(0., -1.,  0.),   //v towards negative Y
		      vec3(-1.,  0., 0.)
        );  //neg X axis
    }
    //YPOS face
	if(face==2u) {
		return array<vec3f, 3>(
		     vec3(1., 0., 0.),     //u towards positive X
		     vec3(0., 0. , -1.),   //v towards negative Z
		     vec3(0., -1. , 0.)
        );  //neg Y axis
    }
    //YNEG face
	if(face==3u) {
		return array<vec3f, 3>(
		     vec3(1., 0., 0.),     //u towards positive X
		     vec3(0., 0., 1.),     //v towards positive Z
		     vec3(0., 1., 0.)
        );   //pos Y axis
    }
    //ZPOS face
	if(face==4u) {
		return array<vec3f, 3>(
		     vec3(1., 0., 0.),     //u towards positive X
		     vec3(0., -1., 0.),    //v towards negative Y
		     vec3(0., 0.,  1.)
        );   //pos Z axis
    }
    //ZNEG face
	if(face==5u) {
		return array<vec3f, 3>(
		     vec3(-1., 0., 0.),    //u towards negative X
		     vec3(0., -1., 0.),    //v towards negative Y
		     vec3(0., 0., -1.)
        );   //neg Z axis
    }

	return array<vec3f, 3>(
		vec3(-0., 0., 0.),    //u towards negative X
		vec3(0., -0., 0.),    //v towards negative Y
		vec3(0., 0., -0.)
	);   //ne
}

fn signed_uv_face_to_cubemap_xyz(uv: vec2f, face_idx: u32) -> vec3f{
	let coords = face_2d_mapping(face_idx);
	// Get current vector
	//generate x,y,z vector (xform 2d NVC coord to 3D vector)
	//U contribution
	let xyz_u = coords[0] * uv.x; // TODO: CP_UDIR but the compiler considers it dynamic indexing and fails
	//V contribution
	let xyz_v = coords[1] * uv.y; // CP_VDIR
	var xyz = xyz_u + xyz_v;
	//add face axis
	xyz = coords[2] + xyz; // CP_FACEAXIS
	//normalize vector
	return normalize(xyz);
}

fn uv_face_to_cubemap_xyz(uv: vec2<f32>, face_idx: u32) -> vec3f {
	let nuv = vec2(uv.x, 1. - uv.y) * 2.0 - vec2(1.0);
	return signed_uv_face_to_cubemap_xyz(nuv, face_idx);
}

fn radicalInverse_VdC(bits: u32) -> f32 {
	var b = bits;
	b = (b << 16u) | (b >> 16u);
	b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
	b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
	b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
	b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
	return f32(b) * 2.3283064365386963e-10; // / 0x100000000
 }

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
fn hammersley(i: u32, n: u32) -> vec2f {
     return vec2(f32(i)/f32(n), radicalInverse_VdC(i));
}

fn importance_sample_ggx(e: vec2f, linear_roughness: f32, n: vec3f ) -> vec3f{
	let m = linear_roughness;

	let phi = 2. * M_PI * e.x;
	let cos_theta = sqrt( (1. - e.y) / ( 1. + (m*m - 1.) * e.y ) );
	let sin_theta = sqrt( 1. - cos_theta * cos_theta );

	let h = vec3(
		sin_theta * cos( phi ),
		sin_theta * sin( phi ),
		cos_theta
	);

	var up_vector = vec3(1.,0.,0.);
	if (abs(n.z) < 0.999) {
		up_vector = vec3(0.,0.,1.);
	}
	let tangent_x = normalize(cross( up_vector, n ));
	let tangent_y = cross( n, tangent_x );
	// tangent to world space
	return normalize(tangent_x * h.x + tangent_y * h.y + n * h.z);
}

fn importance_sample_diffuse(e: vec2f, n: vec3f) -> vec3f {
    let cos_theta = 1.0 - e.y;
    let sin_theta = sqrt(1.0-cos_theta*cos_theta);
	let phi = 2. * M_PI * e.x;

	let h = vec3(
		sin_theta * cos( phi ),
		sin_theta * sin( phi ),
		cos_theta
	);

	var up_vector = vec3(1.,0.,0.);
	if (abs(n.z) < 0.999) {
		up_vector = vec3(0.,0.,1.);
	}
	let tangent_x = normalize( cross( up_vector, n ) );
	let tangent_y = cross( n, tangent_x );

    return tangent_x * h.x + tangent_y * h.y + n * h.z;
}

fn d_ggx(linear_roughness: f32, ndh: f32) -> f32{
    let m = linear_roughness;
    let m2 = m * m;
    let d = (ndh * m2 - ndh) * ndh + 1.0;
    return m2 / (M_PI * d * d);
}

@compute
@workgroup_size(1)
fn radiance(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let resolution = f32(textureDimensions(output_faces).x);
    let texel = vec2<f32>(global_id.xy) / resolution;
    let face = global_id.z;
	let roughness = f32(radiance_data.mip_level) / f32(radiance_data.max_mips - 1u);
	let linear_roughness = roughness * roughness;
    let v = uv_face_to_cubemap_xyz(texel, face);
	let n = v;

	var total_radiance = vec4(0.);
	for(var sample = 0u; sample < MAX_SAMPLES; sample += 1u){
		let xi = hammersley(sample, MAX_SAMPLES);
		let h = importance_sample_ggx(xi, linear_roughness, n);
	    let l = normalize(2. * dot(n, h) * h - n);
		let ndl = max(dot(n, l), 0.);
		let ndh = dot(n, h);
		let hdv = dot(h, v);

		if (ndl > 0.){
			let D = d_ggx(linear_roughness, ndh);
			let pdf = (D * ndh / (4.0 * ndh)) + 0.0001;
			let sa_texel = 4.0 * M_PI / (6.0 * resolution * resolution);
			let sa_sample = 1.0 / (f32(MAX_SAMPLES) * pdf + 0.0001);
			var mip_level = 0.0;
			if (roughness != 0.0) {
				mip_level = 0.5 * log2(sa_sample / sa_texel);
			}
	        let pointRadiance = textureSampleLevel(envmap, envmap_sampler, l, mip_level).rgb;
	        total_radiance += vec4(pointRadiance * ndl, ndl);
	    }
	}

	var color = vec4(0.);
	if (total_radiance.w == 0.){
		color = vec4(total_radiance.rgb, 1.);
	}else{
		color = vec4(total_radiance.rgb / total_radiance.w, 1.);
	}

    textureStore(
		output_faces,
		vec2<i32>(global_id.xy),
		i32(face),
		color
	);
}

@compute
@workgroup_size(1)
fn irradiance(@builtin(global_invocation_id) global_id: vec3<u32>){
	let resolution = f32(textureDimensions(output_faces).x);
    let texel = vec2<f32>(global_id.xy) / resolution;
    let face = global_id.z;
    let v = uv_face_to_cubemap_xyz(texel, face);
	let n = v;

	var total_irradiance = vec4(0.);
	for(var sample = 0u; sample < MAX_SAMPLES; sample += 1u){
		let xi = hammersley(sample, MAX_SAMPLES);
		let h = importance_sample_diffuse(xi, n);
	    let l = normalize(2. * dot(n, h) * h - n);
		let ndl = max(dot(n, l), 0.);

		if (ndl > 0.){
			// Compute Lod using inverse solid angle and pdf.
            // From Chapter 20.4 Mipmap filtered samples in GPU Gems 3.
            // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch20.html
			let pdf = max(0.0, dot(n, l) * M_INV_PI);
			let solidAngleTexel = 4.0 * M_PI / (6.0 * resolution * resolution);
            let solidAngleSample = 1.0 / (f32(MAX_SAMPLES) * pdf + 0.0001);
            let lod = 0.5 * log2(solidAngleSample / solidAngleTexel);

	        let diffuseSample = textureSampleLevel(envmap, envmap_sampler, h, lod).rgb;
	        total_irradiance += vec4(diffuseSample * ndl, ndl);
	    }
	}

	var color = vec4(0.);
	if (total_irradiance.w == 0.){
		color = vec4(total_irradiance.rgb, 1.);
	}else{
		color = vec4(total_irradiance.rgb / total_irradiance.w, 1.);
	}

    textureStore(
		output_faces,
		vec2<i32>(global_id.xy),
		i32(face),
		color
	);

}