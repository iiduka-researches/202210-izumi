# An implementation of the scaled conjugate gradient method

## Usage
### Execution
To conduct experiments, please execute the following: 
#### Constant learning rate
```shell script
python main.py -e <dataset name> -m <model name> -us
#weight decay
python main.py -e <dataset name> -m <model name> -us -wd <weight decay>
```

#### Diminishing learning rate
```shell script
python main.py -e <dataset name> -m <model name>
```
