use vulkanalia::vk;

#[derive(Debug)]
pub struct RawImage {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub pixels: Vec<u8>,
    pub size_bytes: u64,
    pub mip_levels: u32,
}
