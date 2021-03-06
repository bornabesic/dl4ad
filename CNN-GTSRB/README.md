# German Traffic Sign Recognition Benchmark

## Downloading the dataset
```
make dataset
```

## Training

```
usage: train.py [-h] [--learning_rate LEARNING_RATE] [--momentum MOMENTUM]
                [--num_epochs NUM_EPOCHS] [--eval_train EVAL_TRAIN]

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
                        SGD learning rate (default: 0.0001)
  --momentum MOMENTUM   SGD momentum (default: 0.5)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --eval_train EVAL_TRAIN
                        Evaluate the model on the training set after each
                        epoch (default: False)
```

## Evaluating on the test set
```
usage: eval.py [-h] model_path

positional arguments:
  model_path  Path to the saved model parameters file

optional arguments:
  -h, --help  show this help message and exit
```
