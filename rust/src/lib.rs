pub mod error;
pub mod stable_diffusion_interface;

use core::mem::MaybeUninit;
use error::SDError;
use stable_diffusion_interface::*;
use std::path::Path;

const BUF_LEN: i32 = 1000000;

pub type SDResult<T> = Result<T, SDError>;

pub struct Quantization {
    pub model_path: String,
    pub vae_model_path: String,
    pub output_path: String,
    pub wtype: SdTypeT,
}

#[derive(Debug)]
pub enum Task {
    TextToImage,
    ImageToImage,
}

#[derive(Debug)]
pub enum Context<'a> {
    TextToImage(TextToImage<'a>),
    ImageToImage(ImageToImage<'a>),
}

#[derive(Debug)]
pub struct StableDiffusion {
    task: Task,
    model_path: String,
    clip_l_path: String,
    t5xxl_path: String,
    diffusion_model_path: String,
    vae_path: String,
    taesd_path: String,
    control_net_path: String,
    lora_model_dir: String,
    embed_dir: String,
    id_embed_dir: String,
    vae_decode_only: bool,
    vae_tiling: bool,
    n_threads: i32,
    wtype: SdTypeT,
    rng_type: RngTypeT,
    schedule: ScheduleT,
    clip_on_cpu: bool,
    control_net_cpu: bool,
    vae_on_cpu: bool,
}

#[derive(Debug)]
pub struct BaseContext<'a> {
    pub session_id: u32,
    pub prompt: String,
    pub guidance: f32,
    pub width: i32,
    pub height: i32,
    pub control_image: ImageType<'a>,
    pub negative_prompt: String,
    pub clip_skip: i32,
    pub cfg_scale: f32,
    pub sample_method: SampleMethodT,
    pub sample_steps: i32,
    pub seed: i32,
    pub batch_count: i32,
    pub control_strength: f32,
    pub style_ratio: f32,
    pub normalize_input: bool,
    pub input_id_images_dir: String,
    pub canny_preprocess: bool,
    pub upscale_model: String,
    pub upscale_repeats: i32,
    pub output_path: String,
}
pub trait BaseFunction<'a> {
    fn base(&mut self) -> &mut BaseContext<'a>;
    fn set_prompt(&mut self, prompt: &str) -> &mut Self {
        {
            self.base().prompt = prompt.to_string();
        }
        self
    }
    fn set_negative_prompt(&mut self, negative_prompt: impl Into<String>) -> &mut Self {
        {
            self.base().negative_prompt = negative_prompt.into();
        }
        self
    }
    fn set_output_path(&mut self, output_path: &str) -> &mut Self {
        {
            self.base().output_path = output_path.to_string();
        }
        self
    }
    fn generate(&self) -> Result<(), WasmedgeSdErrno>;
}

#[derive(Debug)]
pub struct TextToImage<'a> {
    pub common: BaseContext<'a>,
}

#[derive(Debug)]
pub struct ImageToImage<'a> {
    pub common: BaseContext<'a>,
    pub image: ImageType<'a>,
    pub strength: f32,
}

impl Quantization {
    pub fn new(model_path: &str, output_path: &str, wtype: SdTypeT) -> Quantization {
        Quantization {
            model_path: model_path.to_string(),
            vae_model_path: "".to_string(),
            output_path: output_path.to_string(),
            wtype,
        }
    }
    pub fn convert(&self) -> Result<(), WasmedgeSdErrno> {
        unsafe {
            stable_diffusion_interface::convert(
                &self.model_path,
                &self.vae_model_path,
                &self.output_path,
                self.wtype,
            )
        }
    }
}

impl StableDiffusion {
    pub fn new(task: Task, model_path: &str) -> StableDiffusion {
        let vae_decode_only = match task {
            Task::TextToImage => true,
            Task::ImageToImage => false,
        };
        StableDiffusion {
            task,
            model_path: model_path.to_string(),
            clip_l_path: "".to_string(),
            t5xxl_path: "".to_string(),
            diffusion_model_path: "".to_string(),
            vae_path: "".to_string(),
            taesd_path: "".to_string(),
            control_net_path: "".to_string(),
            lora_model_dir: "".to_string(),
            embed_dir: "".to_string(),
            id_embed_dir: "".to_string(),
            vae_decode_only,
            vae_tiling: false,
            n_threads: -1,
            wtype: SdTypeT::SdTypeCount,
            rng_type: RngTypeT::StdDefaultRng,
            schedule: ScheduleT::DEFAULT,
            clip_on_cpu: false,
            control_net_cpu: false,
            vae_on_cpu: false,
        }
    }

    pub fn new_with_standalone_model(task: Task, diffusion_model_path: &str) -> StableDiffusion {
        let vae_decode_only = match task {
            Task::TextToImage => true,
            Task::ImageToImage => false,
        };
        StableDiffusion {
            task,
            model_path: "".to_string(),
            clip_l_path: "".to_string(),
            t5xxl_path: "".to_string(),
            diffusion_model_path: diffusion_model_path.to_string(),
            vae_path: "".to_string(),
            taesd_path: "".to_string(),
            control_net_path: "".to_string(),
            lora_model_dir: "".to_string(),
            embed_dir: "".to_string(),
            id_embed_dir: "".to_string(),
            vae_decode_only,
            vae_tiling: false,
            n_threads: -1,
            wtype: SdTypeT::SdTypeCount,
            rng_type: RngTypeT::StdDefaultRng,
            schedule: ScheduleT::DEFAULT,
            clip_on_cpu: false,
            control_net_cpu: false,
            vae_on_cpu: false,
        }
    }

    pub fn create_context(&self) -> Result<Context, WasmedgeSdErrno> {
        let mut session_id = MaybeUninit::<u32>::uninit();
        unsafe {
            stable_diffusion_interface::create_context(
                &self.model_path,
                &self.clip_l_path,
                &self.t5xxl_path,
                &self.diffusion_model_path,
                &self.vae_path,
                &self.taesd_path,
                &self.control_net_path,
                &self.lora_model_dir,
                &self.embed_dir,
                &self.id_embed_dir,
                self.vae_decode_only,
                self.vae_tiling,
                self.n_threads,
                self.wtype,
                self.rng_type,
                self.schedule,
                self.clip_on_cpu,
                self.control_net_cpu,
                self.vae_on_cpu,
                session_id.as_mut_ptr(),
            )?;
            let common = BaseContext {
                prompt: "".to_string(),
                session_id: session_id.assume_init(),
                guidance: 3.5,
                width: 512,
                height: 512,
                control_image: ImageType::Path(""),
                negative_prompt: "".to_string(),
                clip_skip: -1,
                cfg_scale: 7.0,
                sample_method: SampleMethodT::EULERA,
                sample_steps: 20,
                seed: 42,
                batch_count: 1,
                control_strength: 0.9,
                style_ratio: 20.0,
                normalize_input: false,
                input_id_images_dir: "".to_string(),
                canny_preprocess: false,
                upscale_model: "".to_string(),
                upscale_repeats: 1,
                output_path: "".to_string(),
            };
            match self.task {
                Task::TextToImage => Ok(Context::TextToImage(TextToImage { common })),
                Task::ImageToImage => Ok(Context::ImageToImage(ImageToImage {
                    common,
                    image: ImageType::Path(""),
                    strength: 0.75,
                })),
            }
        }
    }
    pub fn set_lora_model_dir(&mut self, lora_model_dir: &str) -> &mut Self {
        {
            self.lora_model_dir = lora_model_dir.to_string();
        }
        self
    }
}
impl<'a> BaseFunction<'a> for TextToImage<'a> {
    fn base(&mut self) -> &mut BaseContext<'a> {
        &mut self.common
    }
    fn generate(&self) -> Result<(), WasmedgeSdErrno> {
        if self.common.prompt.is_empty() {
            return Err(WASMEDGE_SD_ERRNO_INVALID_ARGUMENT);
        }
        let mut data: Vec<u8> = vec![0; BUF_LEN as usize];
        let result = unsafe {
            stable_diffusion_interface::text_to_image(
                &self.common.prompt,
                self.common.session_id,
                &self.common.control_image,
                &self.common.negative_prompt,
                self.common.guidance,
                self.common.width,
                self.common.height,
                self.common.clip_skip,
                self.common.cfg_scale,
                self.common.sample_method,
                self.common.sample_steps,
                self.common.seed,
                self.common.batch_count,
                self.common.control_strength,
                self.common.style_ratio,
                self.common.normalize_input,
                &self.common.input_id_images_dir,
                self.common.canny_preprocess,
                &self.common.upscale_model,
                self.common.upscale_repeats,
                &self.common.output_path,
                data.as_mut_ptr(),
                BUF_LEN,
            )
        };
        result?;
        Ok(())
    }
}

impl<'a> BaseFunction<'a> for ImageToImage<'a> {
    fn base(&mut self) -> &mut BaseContext<'a> {
        &mut self.common
    }
    fn generate(&self) -> Result<(), WasmedgeSdErrno> {
        if self.common.prompt.is_empty() {
            return Err(WASMEDGE_SD_ERRNO_INVALID_ARGUMENT);
        }
        match self.image {
            ImageType::Path(path) => {
                if path.is_empty() {
                    return Err(WASMEDGE_SD_ERRNO_INVALID_ARGUMENT);
                }
            }
        }
        let mut data: Vec<u8> = vec![0; BUF_LEN as usize];
        let result = unsafe {
            stable_diffusion_interface::image_to_image(
                &self.image,
                self.common.session_id,
                self.common.guidance,
                self.common.width,
                self.common.height,
                &self.common.control_image,
                &self.common.prompt,
                &self.common.negative_prompt,
                self.common.clip_skip,
                self.common.cfg_scale,
                self.common.sample_method,
                self.common.sample_steps,
                self.strength,
                self.common.seed,
                self.common.batch_count,
                self.common.control_strength,
                self.common.style_ratio,
                self.common.normalize_input,
                &self.common.input_id_images_dir,
                self.common.canny_preprocess,
                &self.common.upscale_model,
                self.common.upscale_repeats,
                &self.common.output_path,
                data.as_mut_ptr(),
                BUF_LEN,
            )
        };
        result?;
        Ok(())
    }
}
impl<'a> ImageToImage<'a> {
    pub fn set_image(&mut self, image: ImageType<'a>) -> &mut Self {
        {
            self.image = image;
        }
        self
    }
    pub fn set_strength(&mut self, strength: f32) -> &mut Self {
        {
            self.strength = strength;
        }
        self
    }
}

#[derive(Debug)]
pub struct SDBuidler {
    sd: StableDiffusion,
}
impl SDBuidler {
    pub fn new(task: Task, model_path: impl AsRef<Path>) -> SDResult<Self> {
        let path = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| SDError::Operation("The model path is not valid unicode.".into()))?;
        let sd = StableDiffusion::new(task, path);
        Ok(Self { sd })
    }

    /// Create a new builder with a full model path.
    pub fn new_with_full_model(task: Task, model_path: impl AsRef<Path>) -> SDResult<Self> {
        let path = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| SDError::Operation("The model path is not valid unicode.".into()))?;
        let sd = StableDiffusion::new(task, path);
        Ok(Self { sd })
    }

    /// Create a new builder with a standalone diffusion model.
    pub fn new_with_standalone_model(
        task: Task,
        diffusion_model_path: impl AsRef<Path>,
    ) -> SDResult<Self> {
        let path = diffusion_model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| SDError::Operation("The model path is not valid unicode.".into()))?;
        let sd = StableDiffusion::new_with_standalone_model(task, path);
        Ok(Self { sd })
    }

    pub fn with_vae_path(mut self, path: impl AsRef<Path>) -> SDResult<Self> {
        let path = path.as_ref().to_str().ok_or_else(|| {
            SDError::Operation("The path to the vae file is not valid unicode.".into())
        })?;
        self.sd.t5xxl_path = path.into();
        Ok(self)
    }

    pub fn with_clip_l_path(mut self, path: impl AsRef<Path>) -> SDResult<Self> {
        let path = path.as_ref().to_str().ok_or_else(|| {
            SDError::Operation("The path to the clip_l file is not valid unicode.".into())
        })?;
        self.sd.clip_l_path = path.into();
        Ok(self)
    }

    pub fn with_t5xxl_path(mut self, path: impl AsRef<Path>) -> SDResult<Self> {
        let path = path.as_ref().to_str().ok_or_else(|| {
            SDError::Operation("The path to the t5xxl file is not valid unicode.".into())
        })?;
        self.sd.t5xxl_path = path.into();
        Ok(self)
    }

    pub fn clip_on_cpu(mut self, enable: bool) -> Self {
        self.sd.clip_on_cpu = enable;
        self
    }

    pub fn vae_on_cpu(mut self, enable: bool) -> Self {
        self.sd.vae_on_cpu = enable;
        self
    }

    pub fn enable_vae_tiling(mut self, enable: bool) -> Self {
        self.sd.vae_tiling = enable;
        self
    }

    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.sd.n_threads = n_threads;
        self
    }
}