import torch
from PIL import Image
from torchvision import transforms as T
from diffusers import StableDiffusionPipeline

from models.main_model import SketchToImageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipe.unet.to(device)
vae = pipe.vae.to(device)

model = SketchToImageModel(unet, vae).to(device)
model.load_state_dict(torch.load("model.pth"))

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

sketch = transform(Image.open("test.png").convert("RGB")).unsqueeze(0).to(device)

latent = torch.randn(1, 4, 64, 64).to(device)
t = torch.tensor([5], device=device)

with torch.no_grad():
    out_latent = model(sketch, latent, t)
    image = vae.decode(out_latent).sample

Image.fromarray((image[0].cpu().numpy().transpose(1,2,0)*255).astype("uint8")).save("output.png")