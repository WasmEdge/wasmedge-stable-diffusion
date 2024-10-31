use core::fmt;
use core::mem::MaybeUninit;
#[repr(transparent)]
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct WasmedgeSdErrno(u32);
pub const WASMEDGE_SD_ERRNO_SUCCESS: WasmedgeSdErrno = WasmedgeSdErrno(0);
pub const WASMEDGE_SD_ERRNO_INVALID_ARGUMENT: WasmedgeSdErrno = WasmedgeSdErrno(1);
pub const WASMEDGE_SD_ERRNO_INVALID_ENCODING: WasmedgeSdErrno = WasmedgeSdErrno(2);
pub const WASMEDGE_SD_ERRNO_MISSING_MEMORY: WasmedgeSdErrno = WasmedgeSdErrno(3);
pub const WASMEDGE_SD_ERRNO_BUSY: WasmedgeSdErrno = WasmedgeSdErrno(4);
pub const WASMEDGE_SD_ERRNO_RUNTIME_ERROR: WasmedgeSdErrno = WasmedgeSdErrno(5);
impl WasmedgeSdErrno {
    pub const fn raw(&self) -> u32 {
        self.0
    }

    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "SUCCESS",
            1 => "INVALID_ARGUMENT",
            2 => "INVALID_ENCODING",
            3 => "MISSING_MEMORY",
            4 => "BUSY",
            5 => "RUNTIME_ERROR",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
    pub fn message(&self) -> &'static str {
        match self.0 {
            0 => "",
            1 => "",
            2 => "",
            3 => "",
            4 => "",
            5 => "",
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }
}
impl fmt::Debug for WasmedgeSdErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmedgeSdErrno")
            .field("code", &self.0)
            .field("name", &self.name())
            .field("message", &self.message())
            .finish()
    }
}
impl fmt::Display for WasmedgeSdErrno {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (error {})", self.name(), self.0)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum SdTypeT {
    SdTypeF32 = 0,
    SdTypeF16 = 1,
    SdTypeQ4_0 = 2,
    SdTypeQ4_1 = 3,
    // SdTypeQ4_2 = 4, // support has been removed
    // SdTypeQ4_3 = 5, // support has been removed
    SdTypeQ5_0 = 6,
    SdTypeQ5_1 = 7,
    SdTypeQ8_0 = 8,
    SdTypeQ8_1 = 9,
    SdTypeQ2K = 10,
    SdTypeQ3K = 11,
    SdTypeQ4K = 12,
    SdTypeQ5K = 13,
    SdTypeQ6K = 14,
    SdTypeQ8K = 15,
    SdTypeIq2Xxs = 16,
    SdTypeIq2Xs = 17,
    SdTypeIq3Xxs = 18,
    SdTypeIq1S = 19,
    SdTypeIq4Nl = 20,
    SdTypeIq3S = 21,
    SdTypeIq2S = 22,
    SdTypeIq4Xs = 23,
    SdTypeI8 = 24,
    SdTypeI16 = 25,
    SdTypeI32 = 26,
    SdTypeI64 = 27,
    SdTypeF64 = 28,
    SdTypeIq1M = 29,
    SdTypeBf16 = 30,
    SdTypeQ4044 = 31,
    SdTypeQ4048 = 32,
    SdTypeQ4088 = 33,
    SdTypeCount = 34,
}
#[derive(Debug, Copy, Clone)]
pub enum RngTypeT {
    StdDefaultRng = 0,
    CUDARng = 1,
}
#[derive(Debug, Copy, Clone)]
pub enum SampleMethodT {
    EULERA = 0,
    EULER = 1,
    HEUN = 2,
    DPM2 = 3,
    DPMPP2SA = 4,
    DPMPP2M = 5,
    DPMPP2Mv2 = 6,
    IPNDM = 7,
    IPNDMV = 8,
    LCM = 9,
}
#[derive(Debug, Copy, Clone)]
pub enum ScheduleT {
    DEFAULT = 0,
    DISCRETE = 1,
    KARRAS = 2,
    EXPONENTIAL = 3,
    AYS = 4,
    GITS = 5,
}
#[derive(Debug)]
pub enum ImageType {
    Path(String),
}
fn parse_image(image: &ImageType) -> (i32, i32) {
    match image {
        ImageType::Path(path) => {
            if path.is_empty() {
                return (0, 0);
            }
            let path = "path:".to_string() + path;
            (path.as_ptr() as i32, path.len() as i32)
        }
    }
}

impl SdTypeT {
    pub fn from_index(index: usize) -> Result<SdTypeT, String> {
        match index {
            0 => Ok(SdTypeT::SdTypeF32),
            1 => Ok(SdTypeT::SdTypeF16),
            2 => Ok(SdTypeT::SdTypeQ4_0),
            3 => Ok(SdTypeT::SdTypeQ4_1),
            // 4 => Ok(SdTypeT::SdTypeQ4_2),// support has been removed
            // 5 => Ok(SdTypeT::SdTypeQ4_3),// support has been removed
            6 => Ok(SdTypeT::SdTypeQ5_0),
            7 => Ok(SdTypeT::SdTypeQ5_1),
            8 => Ok(SdTypeT::SdTypeQ8_0),
            9 => Ok(SdTypeT::SdTypeQ8_1),
            10 => Ok(SdTypeT::SdTypeQ2K),
            11 => Ok(SdTypeT::SdTypeQ3K),
            12 => Ok(SdTypeT::SdTypeQ4K),
            13 => Ok(SdTypeT::SdTypeQ5K),
            14 => Ok(SdTypeT::SdTypeQ6K),
            15 => Ok(SdTypeT::SdTypeQ8K),
            16 => Ok(SdTypeT::SdTypeIq2Xxs),
            17 => Ok(SdTypeT::SdTypeIq2Xs),
            18 => Ok(SdTypeT::SdTypeIq3Xxs),
            19 => Ok(SdTypeT::SdTypeIq1S),
            20 => Ok(SdTypeT::SdTypeIq4Nl),
            21 => Ok(SdTypeT::SdTypeIq3S),
            22 => Ok(SdTypeT::SdTypeIq2S),
            23 => Ok(SdTypeT::SdTypeIq4Xs),
            24 => Ok(SdTypeT::SdTypeI8),
            25 => Ok(SdTypeT::SdTypeI16),
            26 => Ok(SdTypeT::SdTypeI32),
            27 => Ok(SdTypeT::SdTypeI64),
            28 => Ok(SdTypeT::SdTypeF64),
            29 => Ok(SdTypeT::SdTypeIq1M),
            30 => Ok(SdTypeT::SdTypeBf16),
            31 => Ok(SdTypeT::SdTypeQ4044),
            32 => Ok(SdTypeT::SdTypeQ4048),
            33 => Ok(SdTypeT::SdTypeQ4088),
            _ => Ok(SdTypeT::SdTypeCount),
        }
    }
}
impl SampleMethodT {
    pub fn from_index(index: usize) -> Result<SampleMethodT, String> {
        match index {
            0 => Ok(SampleMethodT::EULERA),
            1 => Ok(SampleMethodT::EULER),
            2 => Ok(SampleMethodT::HEUN),
            3 => Ok(SampleMethodT::DPM2),
            4 => Ok(SampleMethodT::DPMPP2SA),
            5 => Ok(SampleMethodT::DPMPP2M),
            6 => Ok(SampleMethodT::DPMPP2Mv2),
            7 => Ok(SampleMethodT::IPNDM),
            8 => Ok(SampleMethodT::IPNDMV),
            _ => Ok(SampleMethodT::LCM),
        }
    }
}
impl ScheduleT {
    pub fn from_index(index: usize) -> Result<ScheduleT, String> {
        match index {
            0 => Ok(ScheduleT::DEFAULT),
            1 => Ok(ScheduleT::DISCRETE),
            2 => Ok(ScheduleT::KARRAS),
            3 => Ok(ScheduleT::EXPONENTIAL),
            4 => Ok(ScheduleT::AYS),
            _ => Ok(ScheduleT::GITS),
        }
    }
}

/// # Safety
/// This will call an external wasm function
pub unsafe fn convert(
    model_path: &str,
    vae_model_path: &str,
    output_path: &str,
    wtype: SdTypeT,
) -> Result<(), WasmedgeSdErrno> {
    let model_path_ptr = model_path.as_ptr() as i32;
    let model_path_len = model_path.len() as i32;
    let vae_model_path_ptr = vae_model_path.as_ptr() as i32;
    let vae_model_path_len = vae_model_path.len() as i32;
    let output_path_ptr = output_path.as_ptr() as i32;
    let output_path_len = output_path.len() as i32;
    let result = wasmedge_stablediffusion::convert(
        model_path_ptr,
        model_path_len,
        vae_model_path_ptr,
        vae_model_path_len,
        output_path_ptr,
        output_path_len,
        wtype as i32,
    );
    if result != 0 {
        Err(WasmedgeSdErrno(result as u32))
    } else {
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
/// # Safety
/// This will call an external wasm function
pub unsafe fn create_context(
    model_path: &str,
    clip_l_path: &str,
    t5xxl_path: &str,
    diffusion_model_path: &str,
    vae_path: &str,
    taesd_path: &str,
    control_net_path: &str,
    lora_model_dir: &str,
    embed_dir: &str,
    id_embed_dir: &str,
    vae_decode_only: bool,
    vae_tiling: bool,
    n_threads: i32,
    wtype: SdTypeT,
    rng_type: RngTypeT,
    schedule: ScheduleT,
    clip_on_cpu: bool,
    control_net_cpu: bool,
    vae_on_cpu: bool,
    session_id: *mut u32,
) -> Result<(), WasmedgeSdErrno> {
    let model_path_ptr = model_path.as_ptr() as i32;
    let model_path_len = model_path.len() as i32;
    let clip_l_path_ptr = clip_l_path.as_ptr() as i32;
    let clip_l_path_len = clip_l_path.len() as i32;
    let t5xxl_path_ptr = t5xxl_path.as_ptr() as i32;
    let t5xxl_path_len = t5xxl_path.len() as i32;
    let diffusion_model_path_ptr = diffusion_model_path.as_ptr() as i32;
    let diffusion_model_path_len = diffusion_model_path.len() as i32;
    let vae_path_ptr = vae_path.as_ptr() as i32;
    let vae_path_len = vae_path.len() as i32;
    let taesd_path_ptr = taesd_path.as_ptr() as i32;
    let taesd_path_len = taesd_path.len() as i32;
    let control_net_path_ptr = control_net_path.as_ptr() as i32;
    let control_net_path_len = control_net_path.len() as i32;
    let lora_model_dir_ptr = lora_model_dir.as_ptr() as i32;
    let lora_model_dir_len = lora_model_dir.len() as i32;
    let embed_dir_ptr = embed_dir.as_ptr() as i32;
    let embed_dir_len = embed_dir.len() as i32;
    let id_embed_dir_ptr = id_embed_dir.as_ptr() as i32;
    let id_embed_dir_len = id_embed_dir.len() as i32;
    let vae_decode_only = vae_decode_only as i32;
    let vae_tiling = vae_tiling as i32;
    let n_threads_ = n_threads;
    let wtype = wtype as i32;
    let rng_type = rng_type as i32;
    let schedule = schedule as i32;
    let clip_on_cpu = clip_on_cpu as i32;
    let control_net_cpu = control_net_cpu as i32;
    let vae_on_cpu = vae_on_cpu as i32;
    let session_id_ptr = session_id as i32;
    let result = wasmedge_stablediffusion::create_context(
        model_path_ptr,
        model_path_len,
        clip_l_path_ptr,
        clip_l_path_len,
        t5xxl_path_ptr,
        t5xxl_path_len,
        diffusion_model_path_ptr,
        diffusion_model_path_len,
        vae_path_ptr,
        vae_path_len,
        taesd_path_ptr,
        taesd_path_len,
        control_net_path_ptr,
        control_net_path_len,
        lora_model_dir_ptr,
        lora_model_dir_len,
        embed_dir_ptr,
        embed_dir_len,
        id_embed_dir_ptr,
        id_embed_dir_len,
        vae_decode_only,
        vae_tiling,
        n_threads_,
        wtype,
        rng_type,
        schedule,
        clip_on_cpu,
        control_net_cpu,
        vae_on_cpu,
        session_id_ptr,
    );
    if result != 0 {
        Err(WasmedgeSdErrno(result as u32))
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
/// # Safety
/// This will call an external wasm function
pub unsafe fn text_to_image(
    prompt: &str,
    session_id: u32,
    control_image: &ImageType,
    negative_prompt: &str,
    guidance: f32,
    width: i32,
    height: i32,
    clip_skip: i32,
    cfg_scale: f32,
    sample_method: SampleMethodT,
    sample_steps: i32,
    seed: i32,
    batch_count: i32,
    control_strength: f32,
    style_ratio: f32,
    normalize_input: bool,
    input_id_images_dir: &str,
    canny_preprocess: bool,
    upscale_model: &str,
    upscale_repeats: i32,
    output_path: &str,
    output_buf: *mut u8,
    out_buffer_max_size: i32,
) -> Result<u32, WasmedgeSdErrno> {
    let prompt_ptr = prompt.as_ptr() as i32;
    let prompt_len = prompt.len() as i32;
    let session_id = session_id as i32;
    let (control_image_ptr, control_image_len) = parse_image(control_image);
    let negative_prompt_ptr = negative_prompt.as_ptr() as i32;
    let negative_prompt_len = negative_prompt.len() as i32;
    let sample_method = sample_method as i32;
    let input_id_images_dir_ptr = input_id_images_dir.as_ptr() as i32;
    let input_id_images_dir_len = input_id_images_dir.len() as i32;
    let normalize_input = normalize_input as i32;
    let canny_preprocess = canny_preprocess as i32;
    let upscale_model_path_ptr = upscale_model.as_ptr() as i32;
    let upscale_model_path_len = upscale_model.len() as i32;
    let output_path_ptr = output_path.as_ptr() as i32;
    let output_path_len = output_path.len() as i32;
    let output_buf_ptr = output_buf as i32;
    let out_buffer_max_size_ = out_buffer_max_size;
    let mut write_bytes = MaybeUninit::<u32>::uninit();
    let result = wasmedge_stablediffusion::text_to_image(
        prompt_ptr,
        prompt_len,
        session_id,
        control_image_ptr,
        control_image_len,
        negative_prompt_ptr,
        negative_prompt_len,
        guidance,
        width,
        height,
        clip_skip,
        cfg_scale,
        sample_method,
        sample_steps,
        seed,
        batch_count,
        control_strength,
        style_ratio,
        normalize_input,
        input_id_images_dir_ptr,
        input_id_images_dir_len,
        canny_preprocess,
        upscale_model_path_ptr,
        upscale_model_path_len,
        upscale_repeats,
        output_path_ptr,
        output_path_len,
        output_buf_ptr,
        out_buffer_max_size_,
        write_bytes.as_mut_ptr() as i32,
    );
    if result != 0 {
        Err(WasmedgeSdErrno(result as u32))
    } else {
        Ok(write_bytes.assume_init())
    }
}
#[allow(clippy::too_many_arguments)]
/// # Safety
/// This will call an external wasm function
pub unsafe fn image_to_image(
    image: &ImageType,
    session_id: u32,
    guidance: f32,
    width: i32,
    height: i32,
    control_image: &ImageType,
    prompt: &str,
    negative_prompt: &str,
    clip_skip: i32,
    cfg_scale: f32,
    sample_method: SampleMethodT,
    sample_steps: i32,
    strength: f32,
    seed: i32,
    batch_count: i32,
    control_strength: f32,
    style_ratio: f32,
    normalize_input: bool,
    input_id_images_dir: &str,
    canny_preprocess: bool,
    upscale_model_path: &str,
    upscale_repeats: i32,
    output_path: &str,
    output_buf: *mut u8,
    out_buffer_max_size: i32,
) -> Result<u32, WasmedgeSdErrno> {
    let (image_ptr, image_len) = parse_image(image);
    let (control_image_ptr, control_image_len) = parse_image(control_image);
    let session_id = session_id as i32;
    let prompt_ptr = prompt.as_ptr() as i32;
    let prompt_len = prompt.len() as i32;
    let negative_prompt_ptr = negative_prompt.as_ptr() as i32;
    let negative_prompt_len = negative_prompt.len() as i32;
    let sample_method = sample_method as i32;
    let normalize_input = normalize_input as i32;
    let input_id_images_dir_ptr = input_id_images_dir.as_ptr() as i32;
    let input_id_images_dir_len = input_id_images_dir.len() as i32;
    let canny_preprocess = canny_preprocess as i32;
    let upscale_model_path_ptr = upscale_model_path.as_ptr() as i32;
    let upscale_model_path_len = upscale_model_path.len() as i32;
    let output_path_ptr = output_path.as_ptr() as i32;
    let output_path_len = output_path.len() as i32;
    let output_buf_ptr = output_buf as i32;
    let out_buffer_max_size_ = out_buffer_max_size;
    let mut write_bytes = MaybeUninit::<u32>::uninit();
    let result = wasmedge_stablediffusion::image_to_image(
        image_ptr,
        image_len,
        session_id,
        guidance,
        width,
        height,
        control_image_ptr,
        control_image_len,
        prompt_ptr,
        prompt_len,
        negative_prompt_ptr,
        negative_prompt_len,
        clip_skip,
        cfg_scale,
        sample_method,
        sample_steps,
        strength,
        seed,
        batch_count,
        control_strength,
        style_ratio,
        normalize_input,
        input_id_images_dir_ptr,
        input_id_images_dir_len,
        canny_preprocess,
        upscale_model_path_ptr,
        upscale_model_path_len,
        upscale_repeats,
        output_path_ptr,
        output_path_len,
        output_buf_ptr,
        out_buffer_max_size_,
        write_bytes.as_mut_ptr() as i32,
    );
    if result != 0 {
        Err(WasmedgeSdErrno(result as u32))
    } else {
        Ok(write_bytes.assume_init())
    }
}
pub mod wasmedge_stablediffusion {
    #[link(wasm_import_module = "wasmedge_stablediffusion")]
    extern "C" {
        pub fn create_context(
            model_path_ptr: i32,
            model_path_len: i32,
            clip_l_path_ptr: i32,
            clip_l_path_len: i32,
            t5xxl_path_ptr: i32,
            t5xxl_path_len: i32,
            diffusion_model_path_ptr: i32,
            diffusion_model_path_len: i32,
            vae_path_ptr: i32,
            vae_path_len: i32,
            taesd_path_ptr: i32,
            taesd_path_len: i32,
            control_net_path_ptr: i32,
            control_net_path_len: i32,
            lora_model_dir_ptr: i32,
            lora_model_dir_len: i32,
            embed_dir_ptr: i32,
            embed_dir_len: i32,
            id_embed_dir_ptr: i32,
            id_embed_dir_len: i32,
            vae_decode_only: i32,
            vae_tiling: i32,
            n_threads: i32,
            wtype: i32,
            rng_type: i32,
            schedule: i32,
            clip_on_cpu: i32,
            control_net_cpu: i32,
            vae_on_cpu: i32,
            session_id_ptr: i32,
        ) -> i32;

        pub fn image_to_image(
            image_ptr: i32,
            image_len: i32,
            session_id: i32,
            guidance: f32,
            width: i32,
            height: i32,
            control_image_ptr: i32,
            control_image_len: i32,
            prompt_ptr: i32,
            prompt_len: i32,
            negative_prompt_ptr: i32,
            negative_prompt_len: i32,
            clip_skip: i32,
            cfg_scale: f32,
            sample_method: i32,
            sample_steps: i32,
            strength: f32,
            seed: i32,
            batch_count: i32,
            control_strength: f32,
            style_ratio: f32,
            normalize_input: i32,
            input_id_images_dir_ptr: i32,
            input_id_images_dir_len: i32,
            canny_preprocess: i32,
            upscale_model_path_ptr: i32,
            upscale_model_path_len: i32,
            upscale_repeats: i32,
            output_path_ptr: i32,
            output_path_len: i32,
            out_buffer_ptr: i32,
            out_buffer_max_size: i32,
            bytes_written_ptr: i32,
        ) -> i32;

        pub fn text_to_image(
            prompt_ptr: i32,
            prompt_len: i32,
            session_id: i32,
            control_image_ptr: i32,
            control_image_len: i32,
            negative_prompt_ptr: i32,
            negative_prompt_len: i32,
            guidance: f32,
            width: i32,
            height: i32,
            clip_skip: i32,
            cfg_scale: f32,
            sample_method: i32,
            sample_steps: i32,
            seed: i32,
            batch_count: i32,
            control_strength: f32,
            style_ratio: f32,
            normalize_input: i32,
            input_id_images_dir_ptr: i32,
            input_id_images_dir_len: i32,
            canny_preprocess: i32,
            upscale_model_path_ptr: i32,
            upscale_model_path_len: i32,
            upscale_repeats: i32,
            output_path_ptr: i32,
            output_path_len: i32,
            out_buffer_ptr: i32,
            out_buffer_max_size: i32,
            bytes_written_ptr: i32,
        ) -> i32;

        pub fn convert(
            model_path_ptr: i32,
            model_path_len: i32,
            vae_model_path_ptr: i32,
            vae_model_path_len: i32,
            output_path_ptr: i32,
            output_path_len: i32,
            wtype: i32,
        ) -> i32;
    }
}
