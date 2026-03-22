import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SketchDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.root = root

        self.sketch_dir = os.path.join(root, "sketches")
        self.image_dir = os.path.join(root, "images")

        self.data = []

        for category in os.listdir(self.sketch_dir):
            sketch_cat = os.path.join(self.sketch_dir, category)
            image_cat = os.path.join(self.image_dir, category)

            for sketch_name in os.listdir(sketch_cat):

                # 🔥 Extract base image name
                base_name = sketch_name.split('-')[0] + ".png"

                image_path = os.path.join(image_cat, base_name)
                sketch_path = os.path.join(sketch_cat, sketch_name)

                # Skip if image does not exist
                if not os.path.exists(image_path):
                    continue

                self.data.append({
                    "sketch": sketch_path,
                    "image": image_path,
                    "category": category
                })

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sketch = Image.open(item["sketch"]).convert("RGB")
        image = Image.open(item["image"]).convert("RGB")

        sketch = self.transform(sketch)
        image = self.transform(image)

        return {
            "sketch": sketch,
            "image": image,
            "category": item["category"]
        }