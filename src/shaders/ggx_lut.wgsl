// Ported from https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/shaders/ibl_filtering.frag
// Copyright Khronos Group
// Apache license 2.0
// Port WGSL Copyright Arturo Castro Prieto

@group(0)
@binding(0)
var lut: texture_storage_2d<rgba32float, write>;

const M_PI = 3.1415926535897932384626433832795;
const NUM_SAMPLES = 128u;


const LAMBERT = 0;
const GGX = 1;


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

// TBN generates a tangent bitangent normal coordinate frame from the normal
// (the normal must be normalized)
fn generate_tbn(normal: vec3f) -> mat3x3<f32> {
    var bitangent = vec3(0.0, 1.0, 0.0);

    let NdotUp = dot(normal, vec3(0.0, 1.0, 0.0));
    let epsilon = 0.0000001;
    if (1.0 - abs(NdotUp) <= epsilon)
    {
        // Sampling +Y or -Y, so we need a more robust bitangent.
        if (NdotUp > 0.0)
        {
            bitangent = vec3(0.0, 0.0, 1.0);
        }
        else
        {
            bitangent = vec3(0.0, 0.0, -1.0);
        }
    }

    let tangent = normalize(cross(bitangent, normal));
    bitangent = cross(normal, tangent);

    return mat3x3(tangent, bitangent, normal);
}

struct MicrofacetDistributionSample {
	phi: f32,
	cos_theta: f32,
	sin_theta: f32,
	pdf: f32,
}

fn d_ggx(linear_roughness: f32, ndh: f32) -> f32{
	let a = ndh * linear_roughness;
    let k = linear_roughness / (1.0 - ndh * ndh + a * a);
    return k * k * (1.0 / M_PI);
}

fn importance_sample_ggx(e: vec2f, linear_roughness: f32, n: vec3f ) -> MicrofacetDistributionSample {
	let m = linear_roughness;

	let phi = 2. * M_PI * e.x;
	let cos_theta = saturate(sqrt( (1. - e.y) / ( 1. + (m*m - 1.) * e.y ) ));
	let sin_theta = sqrt( 1. - cos_theta * cos_theta );

	let pdf = d_ggx(linear_roughness, cos_theta);

	return MicrofacetDistributionSample (
		phi,
		cos_theta,
		sin_theta,
		pdf
	);
}

fn importance_sample_diffuse(e: vec2f, n: vec3f) -> MicrofacetDistributionSample {
	// Cosine weighted hemisphere sampling
    // http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations.html#Cosine-WeightedHemisphereSampling
    let cos_theta = sqrt(1.0 - e.y);
    let sin_theta = sqrt(e.y); // equivalent to `sqrt(1.0 - cosTheta*cosTheta)`;
	let phi = 2. * M_PI * e.x;

	let pdf = cos_theta / M_PI;


	return MicrofacetDistributionSample (
		phi,
		cos_theta,
		sin_theta,
		pdf
	);
}

fn importance_sample(sample: u32, linear_roughness: f32, n: vec3f, distribution: i32) -> vec4f {
	let xi = hammersley(sample, NUM_SAMPLES);
	var importance_sample: MicrofacetDistributionSample;
	if(distribution==LAMBERT) {
		importance_sample = importance_sample_diffuse(xi, n);
	}else if (distribution == GGX) {
		importance_sample = importance_sample_ggx(xi, linear_roughness, n);
	}else{
		// unrecheable
		importance_sample = importance_sample_ggx(xi, linear_roughness, n);
	}


	let h = vec3(
		importance_sample.sin_theta * cos( importance_sample.phi ),
		importance_sample.sin_theta * sin( importance_sample.phi ),
		importance_sample.cos_theta
	);

	let tbn = generate_tbn(n);
	return vec4(tbn * h, importance_sample.pdf);
}

// From the filament docs. Geometric Shadowing function
// https://google.github.io/filament/Filament.html#toc4.4.2
fn v_smith_ggx_correlated(ndv: f32, ndl: f32, linear_roughness: f32) -> f32 {
    let a2 = linear_roughness * linear_roughness;
    let ggxv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let ggxl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / (ggxv + ggxl);
}

// Compute LUT for GGX distribution.
// See https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
@compute
@workgroup_size(1)
fn compute_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let resolution = vec2<f32>(textureDimensions(lut));
	let ndv = (f32(global_id.x)  + 0.5) / resolution.x;
    let roughness = (f32(global_id.y) + 0.5) / resolution.y;
	let linear_roughness = roughness * roughness;

    // Compute spherical view vector: (sin(phi), 0, cos(phi))
    let v = vec3(sqrt(1.0 - ndv * ndv), 0.0, ndv);

    // The macro surface normal just points up.
    let n = vec3(0.0, 0.0, 1.0);

    // To make the LUT independant from the material's F0, which is part of the Fresnel term
    // when substituted by Schlick's approximation, we factor it out of the integral,
    // yielding to the form: F0 * I1 + I2
    // I1 and I2 are slighlty different in the Fresnel term, but both only depend on
    // NoL and roughness, so they are both numerically integrated and written into two channels.
    var a = 0.0;
    var b = 0.0;
    var c = 0.0;

    for(var sample = 0u; sample < NUM_SAMPLES; sample += 1u) {
        // Importance sampling, depending on the distribution.
        let importance_sample = importance_sample(sample, linear_roughness, n, GGX);
        let h = importance_sample.xyz;
        // float pdf = importanceSample.w;
        var l = normalize(reflect(-v, h));

        let ndl = saturate(l.z);
        let ndh = saturate(h.z);
        let vdh = saturate(dot(v, h));
        if (ndl > 0.0) {
			// LUT for GGX distribution.

			// Taken from: https://bruop.github.io/ibl
			// Shadertoy: https://www.shadertoy.com/view/3lXXDB
			// Terms besides V are from the GGX PDF we're dividing by.
			let v_pdf = v_smith_ggx_correlated(ndv, ndl, linear_roughness) * vdh * ndl / ndh;
			let fc = pow(1.0 - vdh, 5.0);
			a += (1.0 - fc) * v_pdf;
			b += fc * v_pdf;
			c += 0.0;
        }
    }

    // The PDF is simply pdf(v, h) -> NDF * <nh>.
    // To parametrize the PDF over l, use the Jacobian transform, yielding to: pdf(v, l) -> NDF * <nh> / 4<vh>
    // Since the BRDF divide through the PDF to be normalized, the 4 can be pulled out of the integral.
    let color = vec3(4.0 * a, 4.0 * b, 4.0 * 2.0 * M_PI * c) / f32(NUM_SAMPLES);
    textureStore(
		lut,
		vec2<i32>(global_id.xy),
		vec4(color, 1.)
	);
}