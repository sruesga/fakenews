# CS 182 Project 1 Proposal

## The Team

### Jihan Yin
I'm a junior EECS major. I've taken CS 189 and EECS 126, and am currently taking CS 182. I've worked on a ML research project in the past for a campus club (Launchpad). 

### Jonathan Lin
I'm a junior EECS major. I've taken CS 189 and EECS 126, and am currently taking CS 182. I've worked on a ML research project in the past for a campus club (Launchpad). 

### Arjun Khare
I'm a junior CS major. I've taken CS 189 and EECS 126, and am currently taking CS 182. I've worked on a ML research project in the past for a campus club (Launchpad). 

### Skyler Ruesga
I'm a junior CS/Stats major. I've taken CS 189, and am currently taking CS 182. I've worked on a ML research project in the past for a campus club (Codebase). 

## Problem Statement and Background

In this day and age, fake news has propogated and dominated social media networks and many other websites through the use of fake accounts and native advertising. The spread of propogandized misinformation has become a serious problem today, influencing elections and destabilizing societies. Currently, fake news is mostly written by humans. However, there is potential for the use of AI to write and mass produce fake news, especially with the rise of sophisticated text generation such as OpenAI's [recent paper](https://blog.openai.com/better-language-models/). As a result, there are a few challenges devoted specifically to detecting fake news (whether written by AI or not), such as the [fake news challenge](http://www.fakenewschallenge.org).

RNNs have been highly effective in generating text, as seen in [this blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). LSTMs are popular variants of RNNs which we see as being better able to take advantage of long-term information. Since generating coherent, understandable text will be the foremost focus of this project, we will likely be using LSTMs (or variants of LSTMs) for this project. Recent work in attentional recurrent models have also shown improvement in areas such as text generation, so we may choose to experiement with attentional models.

We want to explore the potential of this approach so we will be trying our hand at generating fake news.

## Data Source

We intend to use the [Fake News Corpus](https://github.com/several27/FakeNewsCorpus), which is a dataset of about 10 million of news articles scraped from a curated list of data sources. The data is hosted on a public S3 bucket and is about 9 GB in size.

## Description of Tools

We will make a decision on whether to use TensorFlow or PyTorch to build our model after running some benchmark tests. We will also likely utilize some cloud computing resource, e.g. AWS, Google Cloud.

## Evaluation

In running our model, we can choose to either encode words or characters. Many techniques today rely on either one-hot encodings for the words/characters or some sort of learned embedding. With this in mind, we will likely experiment with embedding words using one-hot encodings and also with the popular GLoVe embeddings to train our model. Then, we can apply some sort cross entropy related loss or define another metric in order to evaluate the performance of our model on our dataset, in seeing how accurate we are in predicting individual characters.

We will employ standard techniques to avoid overfitting on our training set, such as cross-validation, dropout, etc. We expect this to also require a lot of human evalution to judge the validity and coherence of the generated statements. 

In addition, we can run a variety of statistical analyses on the generated text to determine coherence and structure of the output. A focus of this project may also be to create new metrics to understand the performance of our model.

