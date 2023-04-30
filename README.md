My solution to the [Kaggle Competition "Contradictory, My Dear Watson"](https://www.kaggle.com/competitions/contradictory-my-dear-watson/overview)

This repository contains a Jupyter notebook that demonstrates the construction of a BERT model for natural language processing (NLP) classification. The model is designed to classify premises and hypotheses as entailment (0), neutral (1), or contradiction (2) - the competition objective.

**Dataset**<br>
The dataset used in this competition consists of pairs of premises and hypotheses, along with their corresponding labels. The training set contains over 120,000 pairs, and the test set contains around 5,000 pairs. The data is provided by Kaggle in CSV format.

**Model Architecture**<br>
The TFBertForSequenceClassification model is a pre-trained deep learning model for NLP classification tasks. In this case, the model is initialized with the pre-trained weights of the bert-base-multilingual-cased model, which is a BERT model that has been trained on multiple languages with cased character input. The num_labels parameter is set to 3 as there are 3 possible labels. The model is trained using the Adam optimizer and the categorical cross-entropy loss function. We use a learning rate of 2e-5 and a batch size of 32.

**Usage**<br>
To use this notebook, you will need to install the following dependencies:
- tensorflow
- transformers
- pandas
- numpy
You will also need to download the competition dataset from [Kaggle](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) and place the train.csv and test.csv files in the data directory.
Once you have installed the dependencies and downloaded the dataset, you can run the notebook to train the model and generate predictions on the test set.

**Conclusion**<br>
This notebook is a simple application of the BERT model for a NLP classification task. BERT models are commonly used by high-rankers in the "Contradictory, My Dear Watson" competition. With further fine-tuning and optimization, it can achieve top notch results on this and other NLP tasks.
