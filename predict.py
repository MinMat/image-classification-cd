import argparse
import torch
import json
from tqdm import tqdm
import data_loader.predict_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from IPython.core.debugger import set_trace


def write_label_predictions(model, device, loader):
    model.eval()
    predictions = {}
    with torch.no_grad():
        set_trace()
        for i, (data, target) in enumerate(tqdm(loader)):
            sample_fname, _ = loader.dataset.samples[i]
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            sample_fname, _ = loader.dataset.samples[i]
           
            #Torch tensor needs to be moved to CPU and then converted to numpy array
            if int(predicted.cpu().data.numpy()[0]) == 0:
                predictions.update({sample_fname: "cat" })
            else:
                predictions.update({sample_fname: "dog"})
            
        return predictions




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
    predictions = write_label_predictions(model, device, data_loader)
    
    #Write predictions to JSON
    with open('predictions.json', 'w', encoding='utf-8') as outfile: 
        outfile.write(json.dumps(predictions, ensure_ascii=False, indent=4))
        #json.dump(predictions, outfile, ensure_ascii=False, indent=4) 
    
 




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
