# Enhancing KL Severity Grading with Focal Loss Optimization and Interpretability through Grad CAM Analysis

## Project Structure
```
.
├── dataset
│   ├── train
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   └── 4
│   ├── val
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   └── 4
│   └── test
│       ├── 0
│       ├── 1
│       ├── 2
│       ├── 3
│       └── 4
├── models
├── custom_densenets.py
├── custom_resnets.py
├── Grad-CAM.ipynb
├── hyperval.py
├── README.md
├── requirements.txt
├── se_nets.py
├── test.ipynb
├── train.py
└── utils.py
```

## Running the files

Dataset can be downloaded from [here.](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

### Training:
Example statement:
```
python3 train.py -m resnet -d dataset -b 32 -l ce -o test -e 100 --learning_rate 1e-3
```
Make sure your path to the dataset is correct. Other parameters can be changed. Do refer to the parser arguments for the same.

### Computing accuracy on Test set, Grad-CAM

To change any of the parameters, go to the notebook and change the parameters in the parser statement

### Hyperparameter Optimization:
Example statement:
```
python3 hyperval.py -m resnet -o test -e 100 -n 100 -s study_name
```
Other parameters can be changed. Do refer to the parser arguments for the same.


## Results
Model weights can downloaded from [here.](https://drive.google.com/drive/folders/1rXnFfJEilQ2eI4Zvh4IyCsa76m0zM2ta?usp=sharing)

| **Architecture**        | **Accuracy** | **Precision** | **Recall** | **F1**   |
|-------------------------|--------------|---------------|------------|----------|
| SE-ResNet18 (CE loss)   | 68.12        | 63.67         | 68.12      | 64.69    |
| SE-DenseNet (CE loss)   | 69.14        | 67.21         | 69.14      | 67.85    |
| SE-ResNet18 (FL)        | 67.75        | 68.56         | 67.75      | 67.83    |
| SE-DenseNet (FL)        | 68.05        | **70.68**      | 68.05      | **68.78**|
| SE-ResNet18 (tuned FL)  | **69.38**    | 67.59         | **69.38**  | 67.75    |

