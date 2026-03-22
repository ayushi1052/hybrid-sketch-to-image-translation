import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline

from dataset import SketchDataset
from models.model import SketchModel
from utils import get_edge, get_depth, get_color

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
vae = pipe.vae.to(device)
unet = pipe.unet.to(device)

model = SketchModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

dataset = SketchDataset("./data")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(3):  # 🔥 only 2–3 epochs
    for batch in loader:

        sketch = batch["sketch"].to(device)
        image = batch["image"].to(device)

        # Encode image
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample() * 0.18215

        # Noise
        noise = torch.randn_like(latents)
        t = torch.randint(0, 50, (latents.size(0),), device=device)

        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

        # Condition
        edge = get_edge(batch["sketch"][0].cpu()).unsqueeze(0).to(device)
        depth = get_depth(batch["sketch"][0].cpu(), device)
        color = get_color(sketch)

        cond = torch.cat([edge, depth, color], dim=1)

        # Forward
        pred = model(noisy_latents, cond)

        loss = loss_fn(pred, latents)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pth")