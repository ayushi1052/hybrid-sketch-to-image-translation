import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline

from models.main_model import SketchToImageModel
from data.dataset import SketchDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
unet = pipe.unet.to(device)
vae = pipe.vae.to(device)

model = SketchToImageModel(unet, vae).to(device)

dataset = SketchDataset("./data")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

for epoch in range(10):
    for sketch, image in loader:
        sketch = sketch.to(device)
        image = image.to(device)

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()

        t = torch.randint(0, 10, (latents.size(0),), device=device)

        output_latent = model(sketch, latents, t)

        loss = loss_fn(output_latent, latents)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")