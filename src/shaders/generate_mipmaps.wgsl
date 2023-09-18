@group(0)
@binding(0)
var input: texture_storage_2d_array<rgba32float, read>;

@group(0)
@binding(1)
var output: texture_storage_2d_array<rgba32float, write>;


@compute
@workgroup_size(1)
fn generate_mipmaps(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let offset = vec2(0, 1);
	let face = i32(global_id.z);
	let out_uv = vec2<i32>(global_id.xy);
	let in_uv = 2 * out_uv;
    let color = (
        textureLoad(input, in_uv + offset.xx, face) +
        textureLoad(input, in_uv + offset.xy, face) +
        textureLoad(input, in_uv + offset.yx, face) +
        textureLoad(input, in_uv + offset.yy, face)
    ) * 0.25;
    textureStore(output, out_uv, face, color);
}
