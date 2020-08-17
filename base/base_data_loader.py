import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from IPython.core.debugger import set_trace


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        

        #This is validaton set
        valid_idx = idx_full[0:len_valid]
        
        # This is training set
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        
        
        test_split = 0.8
        
        if isinstance(test_split, int):
            assert test_split > 0
            assert test_split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_test = test_split
        else:
            len_test = int(len(train_idx) * test_split)
        
        #This is test set
        test_idx = np.delete(train_idx, np.arange(0, len_test))
        updated_train_idx = train_idx[0:len_test]

        train_sampler = SubsetRandomSampler(updated_train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        
      
        return train_sampler, valid_sampler, test_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
    
    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)