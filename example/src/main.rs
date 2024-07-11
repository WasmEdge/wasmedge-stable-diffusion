use wasmedge_stable_diffusion::stable_diffusion_interface::{ImageType, SdTypeT};
use wasmedge_stable_diffusion::{BaseFunction, Context, Quantization, StableDiffusion, Task};
fn main() {
    // if you downloaded ckpt weights, you can use convert() to quantize the ckpt weight to gguf.
    // For running other models, you need to change the model path of the following functions. 
    // let quantization =
    //     Quantization::new("./sd-v1-4.ckpt", "stable-diffusion-v1-4-Q8_0.gguf", SdTypeT::SdTypeQ8_0);
    // quantization.convert().unwrap();
    let context = StableDiffusion::new(Task::TextToImage, "stable-diffusion-v1-4-Q8_0.gguf");
    if let Context::TextToImage(mut text_to_image) = context.create_context().unwrap() {
        text_to_image
            .set_prompt("a lovely cat")
            .set_output_path("output.png")
            .generate()
            .unwrap();
    }
    let context = StableDiffusion::new(Task::ImageToImage, "stable-diffusion-v1-4-Q8_0.gguf");
    if let Context::ImageToImage(mut image_to_image) = context.create_context().unwrap() {
        image_to_image
            .set_prompt("with blue eyes")
            .set_image(ImageType::Path("output.png"))
            .set_output_path("output2.png")
            .generate()
            .unwrap();
    }
}
