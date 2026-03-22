import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class SketchDataset(Dataset):
    def __init__(self, root, size=256):
        self.root = root

        self.sketch_dir = os.path.join(root, "sketch")
        self.photo_dir = os.path.join(root, "photo")

        self.files = []

        for category in os.listdir(self.sketch_dir):
            sketch_cat = os.path.join(self.sketch_dir, category)
            photo_cat = os.path.join(self.photo_dir, category)

            for file in os.listdir(sketch_cat):
                sketch_path = os.path.join(sketch_cat, file)

                # match base name
                base = file.split("-")[0] + ".jpg"
                photo_path = os.path.join(photo_cat, base)

                if os.path.exists(photo_path):
                    self.files.append((sketch_path, photo_path))

        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.files[idx]

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        return self.transform(sketch), self.transform(photo)