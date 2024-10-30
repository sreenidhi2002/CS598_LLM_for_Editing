from diffusers.pipelines import BlipDiffusionPipeline
from diffusers.utils import load_image
import torch

blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
    "Salesforce/blipdiffusion", torch_dtype=torch.float16
).to("cuda")


cond_subject = "dog"
tgt_subject = "dog"
# text_prompt_input = "swimming underwater"
text_prompt_input = "stand on the mirror"
cond_image = load_image(
    "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
)

cond_subject = "backpack"
tgt_subject = "backpack"
# text_prompt_input = "swimming underwater"
# text_prompt_input = "backpack on the floor"s
text_prompt_input = "backpack with flower pattern"
cond_image = load_image(
    "./backpack.png"
)


guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


output = blip_diffusion_pipe(
    text_prompt_input,
    cond_image,
    cond_subject,
    tgt_subject,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    neg_prompt=negative_prompt,
    height=512,
    width=512,
).images
output[0].save(f"blip-{text_prompt_input.replace(' ', '_')}.png")

# save cond_image
cond_image.save("blip-cond.png")
