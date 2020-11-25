# TDT4173-wine-quality
Project repository for predicting wine quality based on physicochemical attributes. The datasets are extracted from https://archive.ics.uci.edu/ml/datasets/wine. 

## Structure of the code
The code builds 3 different models
1. KNN
2. DecisionTree
3. Multi-layer perceptron (MLP)

Furthermore, GridSearchCV is applied to tune hyperparameters. Model performance is then tested on a separate test set. 

The code is structured so that preprocessing techniques (e.g., removal of features) are activated manually. Uncomment to activate, and see code comments for more details.
*Note: PCA is tested in separate code sections, see 'KNN PCA' and 'Tree PCA'*

## How to run the code
Run the code in e.g., a Python notebook or Google colab

To run the different code sections write
```bash
!python main.py -q "[code section name]"
```
