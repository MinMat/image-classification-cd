from torchvision import datasets, transforms
from base import BaseDataLoader
from IPython.core.debugger import set_trace
from torch.utils.data import Dataset
import os
from natsort import natsorted
from PIL import Image


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, idx

class ImageDataLoaderPredict(BaseDataLoader):
    """
     Image data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])  
        self.data_dir = data_dir
        self.dataset = CustomDataSet(self.data_dir, transform=transform)
        #self.dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
