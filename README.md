

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))


## Usage
The training data needs to follow the following folder structure for torch datalaoder to work

```
data/
└── IMG
    ├── test_set
    └── training_set
        ├── cats
        └── dogs
```        


Try `python train.py -c config.json` to train the model .


### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "image_classification",
    "n_gpu": 1,

    "arch": {
        "type": "ImageClassificationModel",
        "args": {}
    },
    
    "data_loader": {
        "type": "ImageDataLoader",
        "args":{
            "data_dir": "data/IMG/training_set",
            "batch_size": 64,
            "shuffle": true,
            "validation_split": 0.2,
            "num_workers": 3
        }
    },  
    "predict_data_loader": {
        "type": "ImageDataLoaderPredict",
        "args":{
            "data_dir": "data/IMG/training_set",
            "batch_size": 1,
            "shuffle": true,
            "validation_split": 0.2,
            "num_workers": 3
        }
    },
    
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001
        }
    },
    "loss": "cross_entropy",
    "metrics": [
        "accuracy"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,

        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        
        "monitor": "min val_loss",
        "early_stop": 10,

        "tensorboard": false
    }
}

```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```
  
### Testing trained models
You can test the model by:

  ```
  python test.py --config config.json --resume path/to/checkpoint
  ```  
Under the results folder you can find the test run results like the visuals for the roc curve


### Running inference from docker 

  Build docker with a tag name from the Docker file. Please execute this command from the project directory "img-classification"
 ```
  docker build -t img-classification .
  ```

  Best model is saved   /saved/models/image_classification/ directory after training

  ``` 
  docker run --shm-size=8g img-classification -c config.json -r "../app/saved/models/image_classification/0820_102734/model_best.pth"
  
  ```

  Copy the prediction json locally. Please replace the path of the docker container with the latest container id after executing docker ps -a 

  ```
  docker cp 2cb9:/app/predictions.json . 

  ```

## License
This project is licensed under the MIT License. See  LICENSE for more details

