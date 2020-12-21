# Unsupervised-Chinese-peotry-text-style-transfer  
## COMS6998 - Practical Deep Learning Systems Performance - Final Project 
### Discription
- This is a model that aim at convert a five-word sentence to a seven-word sentence, and is a implementation of  [A Probabilistic Formulation of Unsupervised Text Style Transfer!](https://arxiv.org/abs/2002.03912)

### Introduction  
- The original data is in folder **data** and the cleaned data is also in it named **five** and **seven**. 
- Run if you want to clean by yourself
```
python getdata_datapreprocess.py
```
- There is pre-trained five-word language model and seven-word language model in folder **model**, you can also train it by yourself by running:

```
python train_lm5.py
```
Or

```
python train_lm5.py
```
And your will get model yourself in folder **model** with name end with *_myself*
- If you want to take a look at the two language models, you can run 

```
python sample_5lm.py
```
Or

```
python sample_7lm.py
```
And the generated sentences will be in folder **output** and named **sampled_five_sentences.txt** or **sampled_seven_sentences.txt**

- If you want to train the VAE model by your self, try
```
python train_vae.py
```
And your will get model yourself in folder **model** with name end with *_myself*
```
python train_vae.py
```
- If you just want to sample a sentence and get the transduction result, run:
```
sample_five2seven.py

```

Or 

```
sample_seven2five.py

```

**However, I have sampled some demo in output folder, you can just download and take a look at it**
