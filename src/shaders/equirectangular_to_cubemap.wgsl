@group(0)
@binding(0)
var equirectangular: texture_storage_2d<rgba32float, read>;

@group(0)
@binding(1)
var cubemap: texture_storage_3d<rgba32float, write>;


const INV_ATAN: vec2<f32> = vec2(0.1591, 0.3183);
const CP_UDIR = 0;
const CP_VDIR = 1;
const CP_FACEAXIS = 2;

fn sample_spherical_map(v: vec3f) -> vec2f
{
    var uv: vec2f = vec2<f32>(atan2(v.z, v.x), asin(v.y));
    uv *= INV_ATAN;
    uv += 0.5;
    return uv;
}

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
	let xyz_u = coords[0] * uv.x; // CP_UDIR
	//V contribution
	let xyz_v = coords[1] * uv.y; // CP_VDIR
	var xyz = xyz_u + xyz_v;
	//add face axis
	xyz = coords[2] + xyz; // CP_FACEAXIS
	//normalize vector
	return normalize(xyz);
}

fn uv_face_to_cubemap_xyz(uv: vec2<u32>, face_idx: u32) -> vec3f {
	let nuv = vec2<f32>(uv) * 2.0 - vec2(1.0);
	return signed_uv_face_to_cubemap_xyz(nuv, face_idx);
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texel = global_id.xy;
    let face = global_id.z;
    let v = uv_face_to_cubemap_xyz(texel, face);
    let uv = vec2<i32>(sample_spherical_map(v) * vec2<f32>(textureDimensions(equirectangular).xy));
    let color = textureLoad(equirectangular, uv);
    textureStore(
		cubemap,
		vec3(vec2<i32>(texel), i32(face)),
		color
		// vec4(vec2<f32>(uv) / vec2<f32>(textureDimensions(equirectangular).xy), 0., 1.)
		//vec4<f32>(vec2<f32>(global_id.xy) / 1024., f32(global_id.z) / 6., 1.));
	);
}
