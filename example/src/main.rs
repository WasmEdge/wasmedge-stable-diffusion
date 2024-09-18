use wasmedge_stable_diffusion::stable_diffusion_interface::{ImageType, SdTypeT, RngTypeT, SampleMethodT, ScheduleT};
use wasmedge_stable_diffusion::{BaseFunction, Context, Quantization, StableDiffusion, Task, SDBuidler};

use clap::{crate_version, Arg, ArgAction, Command};
use std::str::FromStr;
use rand::Rng;
use std::time::{SystemTime, UNIX_EPOCH};

const WTYPE_METHODS: [&str; 35] = [
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "",
    "",
    "q5_0",
    "q5_1",
    "q8_0",
    "q8_1",
    "q2k",
    "q3k",
    "q4k",
    "q5k",
    "q6k",
    "q8k",
    "iq2Xxs",
    "iq2Xs",
    "iq3Xxs",
    "iq1S",
    "iq4N1",
    "iq3S",
    "iq2S",
    "iq4Xs",
    "i8",
    "i16",
    "i32",
    "i64",
    "f64",
    "iq1M",
    "bf16",
    "q4044",
    "q4048",
    "q4088",
    "count"
];
//Sampling Methods
const SAMPLE_METHODS: [&str; 10] = [
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
];

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const SCHEDULE_STR: [&str; 6] = [
    "default",
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("wasmedge-stable-diffusion")
        .version(crate_version!())
        .arg(
            Arg::new("mode")
                .short('M')
                .long("mode")
                .value_name("MODE")
                .value_parser([
                    "txt2img",
                    "img2img",
                    "convert",
                ])
                .help("run mode (txt2img or img2img or convert, default: txt2img).")
                .default_value("txt2img"),
        )
        .arg(
            Arg::new("n_threads")
                .short('t')
                .long("threads")
                .value_parser(clap::value_parser!(i32))
                .value_name("N")
                .help("number of threads to use during computation (default: -1).If threads <= 0, then threads will be set to the number of CPU physical cores.")
                .default_value("-1"),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL")
                .help("path to model.")
                .default_value("stable-diffusion-v1-4-Q8_0.gguf"),
        )
        .arg(
            Arg::new("clip_l_path")
                .long("clip_l")
                .value_name("PATH")
                .help("path to clip_l.")
                .default_value(""),
        )
        .arg(
            Arg::new("t5xxl_path")
                .long("t5xxl")
                .value_name("PATH")
                .help("path to t5xxl.")
                .default_value(""),
        )
        .arg(
            Arg::new("diffusion_model_path")
                .long("diffusion-model")
                .value_name("PATH")
                .help("path to diffusion-model.")
                .default_value(""),
        )
        .arg(
            Arg::new("vae_path")
                .long("vae")
                .value_name("VAE")
                .help("path to vae.")
                .default_value(""),
        )
        .arg(
            Arg::new("taesd_path")
                .long("taesd")
                .value_name("TAESD_PATH")
                .help("path to taesd. Using Tiny AutoEncoder for fast decoding (low quality).")
                .default_value(""),
        )
        .arg(
            Arg::new("control_net_path")
                .long("control-net")
                .value_name("CONTROL_PATH")
                .help("path to control net model.")
                .default_value(""),
        )
        .arg(
            Arg::new("embeddings_path")
                .long("embd-dir")
                .value_name("EMBEDDING_PATH")
                .help("path to embeddings.")
                .default_value(""),
        )
        .arg(
            Arg::new("stacked_id_embd_dir")
                .long("stacked-id-embd-dir")
                .value_name("DIR")
                .help("path to PHOTOMAKER stacked id embeddings.")
                .default_value(""),
        )
        .arg(
            Arg::new("input_id_images_dir")
                .long("input-id-images-dir")
                .value_name("DIR")
                .help("path to PHOTOMAKER input id images dir.")
                .default_value(""),
        )
        .arg(
            Arg::new("normalize_input")
                .long("normalize-input")
                .help("normalize PHOTOMAKER input id images.")
                .action(ArgAction::SetTrue),
        )                        
        .arg(
            Arg::new("upscale_model")
                .long("upscale-model")
                .value_name("ESRGAN_PATH")
                .help("path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.")
                .default_value(""),
        )
        .arg(
            Arg::new("upscale_repeats")
                .long("upscale-repeats")
                .value_parser(clap::value_parser!(i32))
                .value_name("UPSCALE_REPEATS")
                .help("Run the ESRGAN upscaler this many times (default 1).")
                .default_value("1"),
        )
        .arg(
            Arg::new("type")
                .long("type")
                .value_name("TYPE")
                .value_parser([
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                    "q5_0",
                    "q5_1",
                    "q8_0",

                    "q8_1",
                    "q2k",
                    "q3k",
                    "q4k",
                    "q5k",
                    "q6k",
                    "q8k",
                    "iq2Xxs",
                    "iq2Xs",
                    "iq3Xxs",
                    "iq1S",
                    "iq4N1",
                    "iq3S",
                    "iq2S",
                    "iq4Xs",
                    "i8",
                    "i16",
                    "i32",
                    "i64",
                    "f64",
                    "iq1M",
                    "bf16",
                    "q4044",
                    "q4048",
                    "q4088",
                    "count"
                ])
                .help("weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)If not specified, the default is the type of the weight file.")
                .default_value("count"),
        )
        .arg(
            Arg::new("lora_model_dir")
                .long("lora-model-dir")
                .value_name("DIR")
                .help("lora model directory.")
                .default_value(""),
        )
        .arg(
            Arg::new("init_img")
                .short('i')
                .long("init-img")
                .value_name("IMAGE")
                .help("path to the input image, required by img2img.")
                .default_value("./output.png"),
        )             
        .arg(
            Arg::new("control_image")
                .long("control-image")
                .value_name("IMAGE")
                .help("path to image condition, control net.")
                .default_value(""),
        )
        .arg(
            Arg::new("output_path")
                .short('o')
                .long("output")
                .value_name("OUTPUT")
                .help("path to write result image to (default: ./output.png).")
                .default_value("./output2.png"),
        )
        .arg(
            Arg::new("prompt")
                .short('p')
                .long("prompt")
                .value_name("PROMPT")
                .help("the prompt to render.")
                .default_value("a lovely cat"),
        )
        .arg(
            Arg::new("negative_prompt")
                .short('n')
                .long("negative-prompt")
                .value_name("PROMPT")
                .help("the negative prompt.(default: '').")
                .default_value(""),
        )
        .arg(
            Arg::new("cfg_scale")
                .long("cfg-scale")
                .value_parser(clap::value_parser!(f32))
                .value_name("CFG_SCALE")
                .help("unconditional guidance scale: (default: 7.0).")
                .default_value("7.0"),
        )
        .arg(
            Arg::new("strength")
                .long("strength")
                .value_parser(clap::value_parser!(f32))
                .value_name("STRENGTH")
                .help("strength for noising/unnoising (default: 0.75).")
                .default_value("0.75"),
        )
        .arg(
            Arg::new("style_ratio")
                .long("style-ratio")
                .value_parser(clap::value_parser!(f32))
                .value_name("STYLE_RATIO")
                .help("strength for keeping input identity (default: 20%).")
                .default_value("20.0"),
        )
        .arg(
            Arg::new("control_strength")
                .long("control-strength")
                .value_parser(clap::value_parser!(f32))
                .value_name("CONTROL-STRENGTH")
                .help("strength to apply Control Net (default: 0.9) 1.0 corresponds to full destruction of information in init image.")
                .default_value("0.9"),
        )
        .arg(
            Arg::new("guidance")
                .long("guidance")
                .value_parser(clap::value_parser!(f32))
                .value_name("GUAIDANCEE")
                .help("guidance")
                .default_value("3.5"),
        )
        .arg(
            Arg::new("height")
                .short('H')
                .long("height")
                .value_parser(clap::value_parser!(i32))
                .value_name("H")
                .help("image height, in pixel space (default: 512)")
                .default_value("512"),
        )
        .arg(
            Arg::new("width")
                .short('W')
                .long("width")
                .value_parser(clap::value_parser!(i32))
                .value_name("W")
                .help("image width, in pixel space (default: 512)")
                .default_value("512"),
        )
        .arg(
            Arg::new("sampling_method")
                .long("sampling-method")
                .value_parser([
                    "euler_a",
                    "euler",
                    "heun",
                    "dpm2",
                    "dpm++2s_a",
                    "dpm++2m",
                    "dpm++2mv2",
                    "ipndm",
                    "ipndm_v",
                    "lcm",
                ])
                .value_name("SAMPLING_METHOD")
                .help("the sampling method, include values {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm},  sampling method (default: euler_a)")
                .default_value("euler_a"),
        )
        .arg(
            Arg::new("sample_steps")
                .long("steps")
                .value_parser(clap::value_parser!(i32))
                .value_name("STEPS")
                .help("number of sample steps (default: 20).")
                .default_value("20"),
        )           
        .arg(
            Arg::new("rng_type")
                .long("rng")
                .value_name("RNG")
                .value_parser([
                    "std_default",
                    "cuda",
                ])
                .help("RNG (default: std_default).")
                .default_value("std_default"),
        )
        .arg(
            Arg::new("seed")
                .short('s')
                .long("seed")
                .value_parser(clap::value_parser!(i32))
                .value_name("SEED")
                .help("RNG seed (default: 42, use random seed for < 0).")
                .default_value("42"),
        )
        .arg(
            Arg::new("batch_count")
                .short('b')
                .long("batch-count")
                .value_parser(clap::value_parser!(i32))
                .value_name("BATCH_COUNT")
                .help("number of images to generate.")
                .default_value("1"),
        )
        .arg(
            Arg::new("schedule")
                .long("schedule")
                .value_name("SCHEDULE")
                .value_parser([
                    "default",
                    "discrete",
                    "karras",
                    "exponential",
                    "ays",
                    "gits"
                ])
                .help("Denoiser sigma schedule")
                .default_value("default"),
        )
        .arg(
            Arg::new("clip_skip")
                .long("clip-skip")
                .value_parser(clap::value_parser!(i32))
                .value_name("N")
                .help("ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1), <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x.")
                .default_value("-1"),
        )                        
        .arg(
            Arg::new("vae_tiling")
                .long("vae-tiling")
                .help("process vae in tiles to reduce memory usage.")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("control_net_cpu")
                .long("control-net-cpu")
                .help("keep controlnet in cpu (for low vram).")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("canny")
                .long("canny")
                .help("apply canny preprocessor (edge detection).")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("clip_on_cpu")
                .long("clip-on-cpu")
                .help("clip on cpu.")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("vae_on_cpu")
                .long("vae-on-cpu")
                .help("vae on cpu.")
                .action(ArgAction::SetTrue),
        )
        .after_help("run at the dir of .wasm, Example:wasmedge --dir .:. ./target/wasm32-wasi/release/wasmedge_stable_diffusion_example.wasm -m ../../models/stable-diffusion-v1-4-Q8_0.gguf -M img2img\n")
        .get_matches();
    
    //init the paraments--------------------------------------------------------------
    let mut options = Options::default();

    //mode, include "txt2img","img2img",----------"convert" is not yet-------.
    let sd_mode = matches.get_one::<String>("mode").unwrap();
    let task = Task::from_str(sd_mode)?;
    options.mode = sd_mode.to_string();
    
    //n_threads
    let n_threads = matches.get_one::<i32>("n_threads").unwrap();
    options.n_threads = *n_threads as i32;

    //model
    let sd_model = matches.get_one::<String>("model").unwrap();
    options.model_path = sd_model.to_string();
    
    //clip_l_path
    let clip_l_path = matches.get_one::<String>("clip_l_path").unwrap();
    options.clip_l_path = clip_l_path.to_string();

    //t5xxl_path
    let t5xxl_path = matches.get_one::<String>("t5xxl_path").unwrap();
    options.t5xxl_path = t5xxl_path.to_string();

    //diffusion_model_path
    let diffusion_model_path = matches.get_one::<String>("diffusion_model_path").unwrap();
    options.diffusion_model_path = diffusion_model_path.to_string();

    //vae_path
    let vae_path = matches.get_one::<String>("vae_path").unwrap();
    options.vae_path = vae_path.to_string();

    //taesd_path
    let taesd_path = matches.get_one::<String>("taesd_path").unwrap();
    options.taesd_path = taesd_path.to_string();

    //control_net_path
    let control_net_path = matches.get_one::<String>("control_net_path").unwrap();
    options.control_net_path = control_net_path.to_string();

    //embeddings_path
    let embeddings_path = matches.get_one::<String>("embeddings_path").unwrap();
    options.embeddings_path = embeddings_path.to_string();

    //stacked_id_embd_dir
    let stacked_id_embd_dir = matches
        .get_one::<String>("stacked_id_embd_dir").unwrap();
    options.stacked_id_embd_dir = stacked_id_embd_dir.to_string();

    //input_id_images_dir
    let input_id_images_dir = matches
        .get_one::<String>("input_id_images_dir").unwrap();
    options.input_id_images_dir = input_id_images_dir.to_string();

    //normalize-input
    let normalize_input = matches.get_flag("normalize_input");
    options.normalize_input = normalize_input;

    //upscale_model
    let upscale_model = matches.get_one::<String>("upscale_model").unwrap();
    options.upscale_model = upscale_model.to_string();

    //upscale_repeats
    let upscale_repeats = matches.get_one::<i32>("upscale_repeats").unwrap();
    if *upscale_repeats < 1 {
        return Err("Error: the upscale_repeats must be greater than 0".into());
    }
    options.upscale_repeats = *upscale_repeats as i32;

    //type
    let wtype_selected = matches.get_one::<String>("type").unwrap();
    let wtype_found = WTYPE_METHODS
        .iter()
        .position(|&method| method == wtype_selected)
        .ok_or(format!("Invalid wtype: {}",wtype_selected))?;
    let wtype = SdTypeT::from_index(wtype_found)?;
    options.wtype = wtype;

    //lora_model_dir
    let lora_model_dir = matches.get_one::<String>("lora_model_dir").unwrap();
    options.lora_model_dir = lora_model_dir.to_string();

    //init_img, used only for img2img
    let img = matches.get_one::<String>("init_img").unwrap();
    if sd_mode == "img2img" {
        options.init_img = img.to_string();
    };
    
    //control_image
    let control_image = matches.get_one::<String>("control_image").unwrap();
    options.control_image = control_image.to_string();

    //output_path
    let output_path = matches.get_one::<String>("output_path").unwrap();
    options.output_path = output_path.to_string();

    //prompt
    let prompt = matches.get_one::<String>("prompt").unwrap();
    options.prompt = prompt.to_string();

    //negative_prompt
    let negative_prompt = matches.get_one::<String>("negative_prompt").unwrap();
    options.negative_prompt = negative_prompt.to_string();

    //cfg_scale
    let cfg_scale = matches.get_one::<f32>("cfg_scale").unwrap();
    options.cfg_scale = *cfg_scale as f32;

    //strength
    let strength = matches.get_one::<f32>("strength").unwrap();
    if *strength < 0.0  || *strength > 1.0 {
        return Err("Error: can only work with strength in [0.0, 1.0]".into());
    }
    options.strength = *strength as f32;

    //style_ratio
    let style_ratio = matches.get_one::<f32>("style_ratio").unwrap();
    if *style_ratio > 100.0 {
        return Err("Error: can only work with style_ratio in [0.0, 100.0]".into());
    }
    options.style_ratio = *style_ratio as f32;

    //control_strength
    let control_strength = matches.get_one::<f32>("control_strength").unwrap();
    if *control_strength > 1.0 {
        return Err("Error: can only work with control_strength in [0.0, 1.0]".into());
    }
    options.control_strength = *control_strength as f32;

    //guidance
    let guidance = matches.get_one::<f32>("guidance").unwrap();
    options.guidance = *guidance as f32;

    //height
    let height = matches.get_one::<i32>("height").unwrap();
    options.height = *height as i32;

    //width
    let width = matches.get_one::<i32>("width").unwrap();
    options.width = *width as i32;

    //sampling_method
    let sampling_method_selected = matches .get_one::<String>("sampling_method").unwrap();
    let sample_method_found = SAMPLE_METHODS
        .iter()
        .position(|&method| method == sampling_method_selected)
        .ok_or(format!("Invalid sampling method: {}",sampling_method_selected))?;
    let sample_method = SampleMethodT::from_index(sample_method_found)?;
    options.sample_method = sample_method;

    //sample_steps
    let sample_steps = matches.get_one::<i32>("sample_steps").unwrap();
    if *sample_steps <= 0 {
        return Err("Error: the sample_steps must be greater than 0".into());
    }
    options.sample_steps = *sample_steps as i32;

    //rng_type
    let mut rng_type = RngTypeT::StdDefaultRng;
    let rng_type_str = matches.get_one::<String>("rng_type").unwrap();
    if rng_type_str == "cuda"{
        rng_type =  RngTypeT::CUDARng;
    }
    options.rng_type = rng_type;

    //seed
    let seed_str = matches.get_one::<i32>("seed").unwrap();
    let mut seed  = *seed_str;
    // let mut seed: i32 = seed_str.parse().expect("Failed to parse seed as i32");
    if seed < 0 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let current_time_secs = current_time.as_secs() as u32;

        let mut rng = rand::thread_rng();
        seed = ((rng.gen::<u32>() ^ current_time_secs) & i32::MAX as u32) as i32; // 将结果限制在 i32 范围内
    }
    options.seed = seed;

    //batch_count
    let batch_count = matches.get_one::<i32>("batch_count").unwrap();
    options.batch_count = *batch_count as i32;

    //schedule
    let schedule_selected = matches.get_one::<String>("schedule").unwrap();
    let schedule_found = SCHEDULE_STR
        .iter()
        .position(|&method| method == schedule_selected)
        .ok_or(format!("Invalid sampling method: {}",schedule_selected))?;
    // Convert an index to an enumeration value
    let schedule = ScheduleT::from_index(schedule_found)?;
    options.schedule = schedule;
    
    //clip_skip
    let clip_skip = matches.get_one::<i32>("clip_skip").unwrap();
    options.clip_skip = *clip_skip as i32;

    //vae_tiling
    let vae_tiling = matches.get_flag("vae_tiling");
    options.vae_tiling = vae_tiling;
    
    //control_net_cpu
    let control_net_cpu = matches.get_flag("control_net_cpu");
    options.control_net_cpu = control_net_cpu;

    //canny
    let canny = matches.get_flag("canny");
    options.canny = canny;

    //clip_on_cpu
    let clip_on_cpu = matches.get_flag("clip_on_cpu");
    options.clip_on_cpu = clip_on_cpu;

    //vae_on_cpu
    let vae_on_cpu = matches.get_flag("vae_on_cpu");
    options.vae_on_cpu = vae_on_cpu;


    //DEBUG: print options from CL
    print_params(&mut options);
    
    //------------------------------- run the model ----------------------------------------
    // let context = StableDiffusion::new(task, sd_model,
    //     vae_path,
    //     taesd_path, 
    //     control_net_path, 
    //     lora_model_dir, embeddings_path, stacked_id_embd_dir,
    //     vae_tiling, 
    //     *n_threads,
    //     wtype,
    //     rng_type,
    //     schedule,
    //     clip_on_cpu,
    //     control_net_cpu,
    //     vae_on_cpu
    // );
    let context = SDBuidler::new(task, sd_model)?
        .with_clip_l_path("")?
        .with_t5xxl_path("")?
        .with_vae_path(options.vae_path)?
        .with_taesd_path(options.taesd_path)?
        .with_control_net_path(options.control_net_path)?
        .with_lora_model_dir(options.lora_model_dir)?
        .with_embeddings_path(options.embeddings_path)?
        .with_stacked_id_embeddings_path(options.stacked_id_embd_dir)?
        .with_n_threads(options.n_threads)
        .with_wtype(options.wtype)
        .with_rng_type(options.rng_type)
        .with_schedule(options.schedule)
        .enable_vae_tiling(options.vae_tiling)
        .enable_clip_on_cpu(options.clip_on_cpu)
        .enable_control_net_cpu(options.control_net_cpu)
        .enable_vae_on_cpu(options.vae_on_cpu)
        .build();

    match sd_mode.as_str(){
        "txt2img" => {
            println!("txt2img");
            if let Context::TextToImage(mut text_to_image) = context.create_context().unwrap() {
                text_to_image
                    .set_base_params(options.prompt,
                        options.guidance,
                        options.width,
                        options.height,
                        ImageType::Path(&options.control_image),
                        options.negative_prompt,
                        options.clip_skip,
                        options.cfg_scale,
                        options.sample_method,
                        options.sample_steps,
                        options.seed,
                        options.batch_count,
                        options.control_strength,
                        options.style_ratio,
                        options.normalize_input,
                        options.input_id_images_dir,
                        options.canny,
                        options.upscale_model,
                        options.upscale_repeats,
                        options.output_path
                    )
                    .generate()
                    .unwrap();
            }
        },
        "img2img" => {
            println!("img2img");
            if let Context::ImageToImage(mut image_to_image) = context.create_context().unwrap() {
                image_to_image
                    .set_base_params(options.prompt,
                        options.guidance,
                        options.width,
                        options.height,
                        ImageType::Path(&options.control_image),
                        options.negative_prompt,
                        options.clip_skip,
                        options.cfg_scale,
                        options.sample_method,
                        options.sample_steps,
                        options.seed,
                        options.batch_count,
                        options.control_strength,
                        options.style_ratio,
                        options.normalize_input,
                        options.input_id_images_dir,
                        options.canny,
                        options.upscale_model,
                        options.upscale_repeats,
                        options.output_path
                    )
                    .set_image(ImageType::Path(&options.init_img))
                    .set_strength(options.strength)
                    .generate()
                    .unwrap();
            }
        },
        "convert" => {
            println!("into Mode: Convert!");
            // Quantization::new("./sd-v1-4.ckpt", "stable-diffusion-v1-4-Q8_0.gguf", SdTypeT::SdTypeQ8_0);
            let quantization = Quantization::new( sd_model, output_path, wtype);
            quantization.convert().unwrap();
        },
        _ => {
            println!("Error: this mode isn't supported!");
        }
    }

    return Ok(());
}


#[derive(Debug)]
struct Options {
    n_threads: i32,
    mode: String,
    model_path: String,
    clip_l_path: String,
    t5xxl_path: String,
    diffusion_model_path: String,
    vae_path: String,
    taesd_path: String,
    control_net_path: String,
    upscale_model: String,
    embeddings_path: String,
    stacked_id_embd_dir: String,
    input_id_images_dir: String,
    wtype: SdTypeT,
    lora_model_dir: String,
    output_path: String,
    init_img: String,
    control_image: String,


    prompt: String,
    negative_prompt: String,
    cfg_scale: f32,
    guidance: f32,
    style_ratio: f32,
    clip_skip: i32,
    width: i32,
    height: i32,
    batch_count: i32,


    sample_method: SampleMethodT,
    schedule: ScheduleT,
    sample_steps: i32,
    strength: f32,
    control_strength: f32,
    rng_type: RngTypeT,
    seed: i32,
    vae_tiling: bool,
    control_net_cpu: bool,
    normalize_input: bool,
    clip_on_cpu: bool,
    vae_on_cpu: bool,
    canny: bool,
    upscale_repeats: i32, 
}

impl Default for Options {
    fn default() -> Self {
        Self {
            n_threads: -1,
            mode: String::from("txt2img"),
            model_path: String::from(""),
            clip_l_path: String::from(""),
            t5xxl_path: String::from(""),
            diffusion_model_path: String::from(""),
            vae_path: String::from(""),
            taesd_path: String::from(""),
            control_net_path: String::from(""),
            upscale_model: String::from(""),
            embeddings_path: String::from(""),
            stacked_id_embd_dir: String::from(""),
            input_id_images_dir: String::from(""),
            wtype: SdTypeT::SdTypeCount,
            lora_model_dir: String::from(""),
            output_path: String::from(""),
            init_img: String::from(""),
            control_image: String::from(""),
        
            prompt: String::from(""),
            negative_prompt: String::from(""),
            cfg_scale: 7.0,
            guidance: 3.5,
            style_ratio: 20.0,
            clip_skip: -1,
            width: 512,
            height: 512,
            batch_count: 1,
        
            sample_method: SampleMethodT::EULERA,
            schedule: ScheduleT::DEFAULT,
            sample_steps: 20,
            strength: 0.75,
            control_strength: 0.9,
            rng_type: RngTypeT::StdDefaultRng,
            seed: 42,
            vae_tiling: false,
            control_net_cpu: false,
            normalize_input: false,
            clip_on_cpu: false,
            vae_on_cpu: false,
            canny: false,
            upscale_repeats: 1, 
        }
    }
}

fn print_params(params: &mut Options) {
    println!("Option:");
    println!("[INFO] n_threads:         {}", params.n_threads);
    println!("[INFO] mode:              {}", params.mode);
    println!("[INFO] model_path:        {}", params.model_path);
    println!("[INFO] clip_l_path:       {}", params.clip_l_path);
    println!("[INFO] t5xxl_path:        {}", params.t5xxl_path);
    println!("[INFO] diffusion_model_path:{}", params.diffusion_model_path);
    println!("[INFO] vae_path:          {}", params.vae_path);
    println!("[INFO] taesd_path:        {}", params.taesd_path);
    println!("[INFO] control_net_path:  {}", params.control_net_path);
    println!("[INFO] upscale_model:     {}", params.upscale_model);
    println!("[INFO] embeddings_path:   {}", params.embeddings_path);
    println!("[INFO] stacked_id_embd:   {}", params.stacked_id_embd_dir);
    println!("[INFO] input_id_images:   {}", params.input_id_images_dir);
    println!("[INFO] wtype:             {:?}", params.wtype);
    println!("[INFO] lora_model_dir:    {}", params.lora_model_dir);
    println!("[INFO] output_path:       {}", params.output_path);
    println!("[INFO] init_img:          {}", params.init_img);
    println!("[INFO] control_image:     {}", params.control_image);
    println!("[INFO] prompt:            {}", params.prompt);
    println!("[INFO] negative_prompt:   {}", params.negative_prompt);
    println!("[INFO] cfg_scale:         {}", params.cfg_scale);
    println!("[INFO] guidance:          {}", params.guidance);
    println!("[INFO] style_ratio:       {}", params.style_ratio);
    println!("[INFO] clip_skip:         {}", params.clip_skip);
    println!("[INFO] width:             {}", params.width);
    println!("[INFO] height:            {}", params.height);
    println!("[INFO] batch_count:       {}", params.batch_count);
    println!("[INFO] sample_method:     {:?}", params.sample_method);
    println!("[INFO] schedule:          {:?}", params.schedule);
    println!("[INFO] sample_steps:      {}", params.sample_steps);
    println!("[INFO] strength:          {}", params.strength);
    println!("[INFO] control_strength:  {}", params.control_strength);
    println!("[INFO] rng_type:          {:?}", params.rng_type);
    println!("[INFO] seed:              {}", params.seed);
    println!("[INFO] vae_tiling:        {}", params.vae_tiling);
    println!("[INFO] control_net_cpu:   {}", params.control_net_cpu);
    println!("[INFO] normalize_input:   {}", params.normalize_input);
    println!("[INFO] clip_on_cpu:       {}", params.clip_on_cpu);
    println!("[INFO] vae_on_cpu:        {}", params.vae_on_cpu);
    println!("[INFO] canny:             {}", params.canny);
    println!("[INFO] upscale_repeats:   {}", params.upscale_repeats);
}
