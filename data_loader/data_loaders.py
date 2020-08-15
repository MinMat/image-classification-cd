from torchvision import datasets, transforms
from base import BaseDataLoader
from IPython.core.debugger import set_trace

     
class ImageDataLoader(BaseDataLoader):
    """
     Image data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])  
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
