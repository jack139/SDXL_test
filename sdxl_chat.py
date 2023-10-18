import argparse
import time
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=".", type=str)
parser.add_argument('--no_refine', action='store_true', help='do not use refine model')
parser.add_argument('--no_compile', action='store_true', help='do not compile model (PyTorch < 2.0)')
args = parser.parse_args()

'''
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
'''

start = time.time()
print("Load model base model ...")
base = DiffusionPipeline.from_pretrained(f"{args.model_path}/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
if not args.no_compile:
    base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
print(f"Time elapsed: {(time.time() - start):.3f} sec.")

if not args.no_refine:
    start = time.time()
    print("Load model base model ...")
    refiner = DiffusionPipeline.from_pretrained(
        f"{args.model_path}/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    if not args.no_compile:
        refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
    print(f"Time elapsed: {(time.time() - start):.3f} sec.")



def gen(prompt, refine=False):
    if refine:
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
    else:
        image = base(prompt=prompt).images[0]
    
    image.save(f"sdxl-{int(time.time())}.jpg")


if __name__ == '__main__':
    with torch.no_grad():
        print("Start inference mode.")
        print('=' * 85)

        while True:
            raw_input_text = input("Prompt:")
            raw_input_text = str(raw_input_text)
            if len(raw_input_text.strip()) == 0:
                break

            start = time.time()
            gen(raw_input_text, not args.no_refine)
            print(f">>>>>>> Time elapsed: {(time.time() - start):.3f} sec.\n\n")


'''
# 单独使用 refine
pipe2 = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "./stable-diffusion-xl-refiner-1.0", torch_dtype=torch.bfloat16, variant="bfp16", use_safetensors=True
).to("cuda")
pipe2.unet = torch.compile(pipe2.unet, mode="reduce-overhead", fullgraph=True)

image1 = load_image("1.png").convert("RGB")

prompt = "a photo of an astronaut riding a horse on mars"
imag2 = pipe2(prompt, image=image1).images[0]

image2.save("tmp2.jpg")
'''