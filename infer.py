import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from models.model import SketchModel
from utils import get_edge, get_depth, get_color

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler

model = SketchModel().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

sketch = Image.open("test.png").convert("RGB")

edge = get_edge(sketch).unsqueeze(0).to(device)
depth = get_depth(sketch, device)
sketch_tensor = torch.randn(1,3,256,256).to(device)  # placeholder
color = get_color(sketch_tensor)

cond = torch.cat([edge, depth, color], dim=1)

latent = torch.randn(1, 4, 32, 32).to(device)

scheduler.set_timesteps(20)

for t in scheduler.timesteps:

    with torch.no_grad():
        noise_pred = unet(latent, t).sample

    latent = model(latent, cond)
    latent = scheduler.step(noise_pred, t, latent).prev_sample

image = vae.decode(latent / 0.18215).sample

image = (image[0].cpu().permute(1,2,0).numpy() * 255).clip(0,255).astype("uint8")
Image.fromarray(image).save("output.png")