import matplotlib.pyplot as plt

import argparse
import torch
import os

from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import numpy as np

from torch import nn
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix,f1_score,auc,roc_curve


from IPython.core.debugger import set_trace

def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class].cpu().numpy()))
    return [i.item() for i in actuals], [i.item() for i in probabilities]



def predict_image(image, model, dataloader, device):
    image_tensor = dataloader.dataset.transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    sm = torch.nn.Softmax()
    probabilities = sm(output)
    return probabilities

def get_random_images(num,dataloader):
    data = dataloader.dataset
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


def main(config):
    
    logger = config.get_logger('test')
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader =  data_loader.split_test() 
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    
    #Model Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    actuals, predictions = test_label_predictions(model, device, test_data_loader)
    
    print('Confusion matrix:')
    print(confusion_matrix(actuals, predictions))
    print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
  
    
    which_class = 1
    actuals, class_probabilities = test_class_probabilities(model, device, test_data_loader, which_class)
    
   
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)   
    print('roc_auc score: %f' % roc_auc)
    
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    sample_file_name = "sample"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for digit=%d class' % which_class)
    plt.legend(loc="lower right")
    plt.savefig(results_dir + sample_file_name)
   
    
    
    
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
