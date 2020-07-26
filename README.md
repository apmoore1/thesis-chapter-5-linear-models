# thesis-chapter-5-linear-models
Reproduction study of TDParse and Target Dep


## Data

### Datasets

1. [Dong et al. 2014](https://www.aclweb.org/anthology/P14-2009/) Twitter dataset. [Training](https://raw.githubusercontent.com/bluemonk482/tdparse/master/data/lidong/training/target.train.raw.raw) and [test](https://raw.githubusercontent.com/bluemonk482/tdparse/master/data/lidong/testing/target.test.raw.raw) datasets can be found at their relevant links. The datasets are then downloaded to [./data/dong/target.train.raw.raw](./data/dong/target.train.raw.raw) and [./data/dong/target.test.raw.raw](./data/dong/target.test.raw.raw) respectively.
2. [Wang et al. 2017](https://www.aclweb.org/anthology/E17-1046/) Election Twitter dataset. [Data folder](https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1) that is to be downloaded to [./data/election](./data/election) and all tar/zipped files to be untarred/unzipped so that they become folders.
3. [Mitchell et al. 2013](https://www.aclweb.org/anthology/D13-1171.pdf) English Twitter dataset [link here](https://www.m-mitchell.com/code/MitchellEtAl-13-OpenSentiment.tgz). Ensure the dataset has been downloaded and uncompressed to [./data/mitchell](./data/mitchell). After which run the through the following [notebook](./mitchell_notebook.ipynb) to create the train and test splits used here. This notebook is a copy of [this](https://github.com/apmoore1/Bella/blob/master/notebooks/Mitchel%20et%20al%20dataset%20splitting.ipynb).
4. YouTuBean from [Marrese-Taylor et al. 2017](https://www.aclweb.org/anthology/W17-5213), of which the data can be found [here](https://raw.githubusercontent.com/epochx/opinatt/master/samsung_galaxy_s5.xml) and download to [./data/samsung_galaxy_s5.xml](./data/samsung_galaxy_s5.xml). After which run through the following [notebook](./youtubean.ipynb) to create the train and test splits used here. This notebook is a copy of [this](https://github.com/apmoore1/Bella/blob/master/notebooks/YouTuBean%20dataset%20splitting.ipynb).
5. SemEval 2014 Laptop dataset by [Pontiki et al. 2014](https://www.aclweb.org/anthology/S14-2004.pdf). The training data can be found [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and test [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/). After decompressing the data should be found at [./data/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml](./data/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml) and [./data/ABSA_Gold_TestData/Laptops_Test_Gold.xml](./data/ABSA_Gold_TestData/Laptops_Test_Gold.xml) respectively.
6. SemEval 2014 Restaurant dataset by [Pontiki et al. 2014](https://www.aclweb.org/anthology/S14-2004.pdf). This data was downloaded when downloading the laptop data. Thus the train and test can be found [./data/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml](./data/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml) and [./data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml](./data/ABSA_Gold_TestData/Restaurants_Test_Gold.xml) respectively.

For the neural models the following script needs running to convert the datasets into the correct format for these models as well as to split the training split into train and validation splits for early stopping. This data will then be saved to the following directory [./data/neural](./data/neural)

``` bash
python create_train_val_test.py
```

Creating the small training datasets for the mass evaluation can be done by using the following script:

``` bash
python create_small_training_sets.py
```

### Sentiment lexicons

1. MPQA [Wilson et al. 2005](https://www.aclweb.org/anthology/H05-1044/), which can be downloaded from [here](https://mpqa.cs.pitt.edu/lexicons/subj_lexicon/) and will be stored [./data/lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff](./data/lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff).
2. NRC [Mohammad and Turney 2010](https://www.aclweb.org/anthology/W10-0204/), which can be downloaded using the following [link](http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip) and will be stored [./data/lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt](./data/lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt)
3. HL [Hu and Liu 2004](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf), which can be downloaded from [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) and will be stored [./data/lexicons/opinion-lexicon-English](./data/lexicons/opinion-lexicon-English). **NOTE** if running a Linux based OS will require `apt install unrar` to un-compress the Hu and Liu sentiment lexicons.

## Notebooks

The reproduction of [Vo et al. 2015 paper](https://ijcai.org/Proceedings/15/Papers/194.pdf) can be found [./reproduction_vo.ipynb](./reproduction_target_dependent.ipynb)


The reproduction of [Wang et al. 2017 paper](https://www.aclweb.org/anthology/E17-1046/) can be found [./reproduction_wang.ipynb](./reproduction_wang.ipynb)

The mass evaluation of [Vo et al. 2015 paper](https://ijcai.org/Proceedings/15/Papers/194.pdf) and [Wang et al. 2017 paper](https://www.aclweb.org/anthology/E17-1046/) methods can be found [./linear_models_mass_evaluation.ipynb](./linear_models_mass_evaluation.ipynb) (all this notebook does is generate the results). The mass evaluation of [Tang et al. 2016](https://www.aclweb.org/anthology/C16-1311.pdf) methods can be found [./tang_mass_evaluation.ipynb](./tang_mass_evaluation.ipynb) (again all this notebook does is generate the results).

### Word Embeddings

For all the Neural Pooling methods of [Vo et al. 2015 paper](https://ijcai.org/Proceedings/15/Papers/194.pdf) and [Wang et al. 2017 paper](https://www.aclweb.org/anthology/E17-1046/) their embeddings are automatically downloaded and saved in the following directory `~/.Bella/Vectors` and normally in a binary format for quick read access. Some of the same word vectors are used in the LSTM approach of [Tang et al. 2016](https://www.aclweb.org/anthology/C16-1311.pdf). However all of these word vectors need to be downloaded manually, thus the following need to be saved in this directory `./word embeddings`:

1. [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/)
2. Glove Twitter 50, 100, and 200 dimension which can be found [here](https://nlp.stanford.edu/projects/glove/)
3. SSWE, which can be downloaded from [here](https://raw.githubusercontent.com/bluemonk482/tdparse/master/resources/wordemb/sswe/sswe-u.txt)

After downloading the SSWE embedding run the following script to re-format the embeddings so that they are in the correct format for the LSTM models:

``` bash
python re_format_sswe.py
```

All the script does is change the space unit from a tab (\t) to a single whitespace token between the vector floats (parameters) on each line.

## Requirements

Python >= 3.6.1

For the LSTM based models we used the following Pytorch version:

`pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

Which is version `1.5.1` I think using CUDA version `10.1`. Before installing the other requirements is the best time to install Pytorch if you want a specific version installed e.g. CPU or GPU version but it has to to be `>=1.5.0,<1.6.0`.

`pip install -r requirements.txt`

**NOTE** To run any of the *TDParse* e.g. [Wang et al. 2017 paper](https://www.aclweb.org/anthology/E17-1046/) models/methods it does require running the following TweeboParser API server through docker:

`docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker`

To use any of the Stanford tools through the `bella` package it does require the Stanford CoreNLP API server through docker:

`docker run -p 9000:9000 -d --rm mooreap/corenlp`