# SmartTab
Google ML Winter Camp Beijing Site 2019 Group Project - 写的代码都队  

## SmartTab
SmartTab aims to automatically analyze the structure of a relatively long article/passage, it takes in the raw article and split it into resonable sections.

## Dependency

| **Package**        | **version** |
|----------------|---------|
| python         | 3.6.8   |
| tensorflow-gpu | 1.12    |
| keras          | 2.2.4   |
| keras-contrib  | -       |
| nltk           | 3.4     |
| FastText       | -       |

### Method
We model the problem as a Sequence Labeling problem that takes 2 steps:

* **Paragraph embedding**: Use BERT as an encoder to encode paragraphs. We have only tested the fixed pre-trained version of BERT model. It is very likely that fine-tuned BERT will perform better. Due to lack of time, we leave it as future work.
* **Sequence Labeling**: We tag each paragraph according to its position within the section it belongs to and the relation between it with the neighbor above. More specifically, there are 3 types of tags: B, M and E, which means beginning, middle, end of a section. For tag B, it also has several variants B{int}, where {int} is an integer represents the depth of the paragraph minus the one above it.

### Model
**BERT** encoder(fixed) + bi-LSTM + CRF to do sequence labeling.

![](https://github.com/My-code-works/SmartTab/blob/master/model.png)

### Code
This repo is a little messy right now, primarily contains:
* ```BERT-LSTM.ipynb```: Training and testing notebook for the currently best-perform model (BERT(fixed) + bi-LSTM + CRF);
* ```bert/```: The pretrained **BERT** model(**BERT-Base, Uncased**) and codes from <https://github.com/google-research/bert>;
* ```LSTMmodel.ipynb```: Training and testing notebook for non-BERT based models;
* ```load-feature.ipynb```: Transfer the original training texts into features using BERT and store locally using ```Pickle``` (otherwise it's to slow to do so during training, ```Pickle``` files are too large to upload into this repo);
* ```*.sh```: Shell scripts to prepare data and models;
* Other ```*.py``` files as helper/utils modules.

Due to the limitation of size, no saved model weights are uploaded into this repo.

### Data
Training data is derived from **WikiJoin** dataset, which contains 114,975 paired articles of the same topic from Wikipedia, in JSON format, the “full” version of the data contain structured section texts from the article.

The Data files are too large to upload. The data folder structure is :

* data/
	* 0.json
	* 1.json
	* ... 

### Experiment Result
| Model | Sequence Labeling Test Accuracy |
| ------ | ------ |
| FastText(mean) + bi-LSTM + CRF | ~ 30% |
| FastText + CNN + bi-LSTM + CRF | ? |
| **BERT(fixed) + bi-LSTM + CRF** | **73.6%** |
| BERT(finetune) + bi-LSTM + CRF | ? |
