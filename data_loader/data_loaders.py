from torchvision import datasets, transforms
from base import BaseDataLoader
from IPython.core.debugger import set_trace


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
        
        
class ImageDataLoader(BaseDataLoader):
    """
     Image data loading using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = {
                'train': transforms.Compose([
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                'val': transforms.Compose([
                    transforms.Resize([224,224]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                }
        
        set_trace()
        self.data_dir = data_dir + "IMG/training_set"
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)