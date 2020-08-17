import argparse
import torch
from tqdm import tqdm
import data_loader.predict_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from IPython.core.debugger import set_trace


def write_label_predictions(model, device, test_loader):
    model.eval()
    predictions = []
    
    f = open("test_y", "w")
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            sample_fname, _ = test_loader.dataset.samples[i]
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            sample_fname, _ = test_loader.dataset.samples[i]
            f.write("{}, {}\n".format(sample_fname, predicted))
    f.close()        


def main(config):
    logger = config.get_logger('predict')
    
        
    # setup data_loader for test instances
    # setup data_loader instances
    data_loader = config.init_obj('predict_data_loader', module_data)
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
    
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    write_label_predictions(model, device, test_data_loader)
 




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
