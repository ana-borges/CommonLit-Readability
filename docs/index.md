![books](https://user-images.githubusercontent.com/38925412/124011275-21ecc980-d9e0-11eb-83ff-c77552a45e00.jpg)

# CommonLit Readability Prize: Rate the complexity of literary passages

This is an exposition of the work in progress for the Kaggle competition [CommonLit Readability Prize](https://www.kaggle.com/c/commonlitreadabilityprize).

So far we have tested Support Vector Machines, XGBoost, Random Forest, GLM, and have started to implement BERT-like models.

## Challenge description

In the competition we have to build a model to rate the reading complexity of literary passages. This literary passages are from texts used in classes of ages ranging from 3 to 12 years old.

## Data description

The data provided by the competition is a single dataset containing the following variables: `id`. `url_legal`, `license`, `excerpt`,`target` and `standard_error`.

From these variables, the only ones containing useful information are `excerpt`,`target` and `standard_error`. They also contain zero missing values.

Let's see what a random `excerpt` is like:

> When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.\nThe floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.\nAt each end of the room, on the wall, hung a beautiful bear-skin rug.\nThese rugs were for prizes, one for the girls and one for the boys. And this was the game.\nThe girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.\nThis would have been an easy matter, but each traveller was obliged to wear snowshoes.

The `target` is the reading difficulty. It is the result of a [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley-Terry_model) analysis of more than 111,000 pairwise comparisons between excerpts. Teachers spanning grades 3-12 (a majority teaching between grades 6−10) served as the raters for these comparisons.

The `standard_error` is included as an output of the Bradley-Terry analysis because individual raters saw only a fraction of the excerpts, while every excerpt was seen by numerous raters. 

On average, each excerpt has been reviewed 22 times.

## Text treatment

The data treatment we have carried out involves the following steps:

* Make the words lower case.
* Remove stop words
* Tokenize words: do this to words
* Remove punctuation
* Stemmatize

After going through this process, an expert such as the one shown before looks like this:

> young peopl return ballroom present decid chang appear instead interior scene winter landscap floor cover snowwhit canva laid smooth rumpl bump hillock like real snow field numer palm evergreen decor room powder flour strewn tuft cotton like snow also diamond dust light sprinkl them glitter crystal icicl hung branch end room wall hung beauti bearskin rug rug prize one girl one boy game girl gather one end room boy other one end call north pole south pole player given small flag plant reach pole would easi matter travel oblig wear snowsho'


## Text measurement

In order to grasp a better understanding of the complexity of the texts we have tried several measures and see how they correlate with the reported reading complexity for the texts.

We can separate this measures in two kinds: text transformations and readability scores.

### Text Transformations

These are de measures we tested so far:

* Length of the text (processed and unprocessed): No correlation with `target` found.
* Mean length of the words (processed and unprocessed): No correlation with `target` found.
* Number of words erased by processing the text: No correlation with `target` found.

As an example, see the graph below showing the plot of number of erased words vs `target`.

![words erased](https://user-images.githubusercontent.com/38925412/124133364-0d610d80-da82-11eb-95e3-cb26540fd3a8.png)

### Readability Scores

The measures we have tried, using the [textstat](https://pypi.org/project/textstat/) library are the following:

* [Dale–Chall readability formula](https://en.wikipedia.org/wiki/Dale-Chall_readability_formula)
* [Flesch-Kincaid grade level](https://en.wikipedia.org/wiki/Flesch-Kincaid_readability_tests#Flesch-Kincaid_grade_level)
* [Flesch reading ease](https://en.wikipedia.org/wiki/Flesch-Kincaid_readability_tests#Flesch_reading_ease)
* [Coleman-Liau index](https://en.wikipedia.org/wiki/Coleman-Liau_index)
* [Automated readability index](https://en.wikipedia.org/wiki/Automated_readability_index)
* [Gunning fog index](https://en.wikipedia.org/wiki/Gunning_fog_index)
* [Linsear write](https://en.wikipedia.org/wiki/Linsear_Write)
* [Readability Consensus based upon all the above tests](https://pypi.org/project/textstat/)
* Punctuation count
* Punctuation score
* Lexicon count
* Lexicon score
* Sentence count
* Sentence score

To see how they correlate with each other, `target` and `standard_error` we refer to the below image.

![heatmap](https://user-images.githubusercontent.com/38925412/124137557-148a1a80-da86-11eb-8d47-c8ae178c5886.png)

## Model evaluation

In this phase we have started with model selection. So far we have tried Support Vector Machines, XGBoost regressor, Random Forest, GLM.

### Support Vector Machines

We have tried out two kind of SVM: Support Vector Regressor and Nu Support Vector Regressor, each with different kernels.

#### SVR

We tried the `linear`, `poly` and `rfb` kernels.

The mean squared error for the kernels is

* `linear` MSE: 0.4482325965884177
* `poly` MSE: 0.700934980309811
* `rfb` MSE: 0.7032331820275759

#### NuSVR

We tried the `linear`, `poly` and `rbf` kernels.

The mean squared error for the kernel is

* `linear` MSE: 0.4328520467681354
* `poly` MSE: 0.705854310605194
* `rbf` MSE: 0.7071887919114409

### XGBoost regressor

For the XGBoost regressor the implementation was quite straightforward, yielding the following MSE:

* MSE: 0.6471750555617857

### Random forest

For the random forest we got the following measures:

* Mean Absolute Error (MAE): 0.5254135828139506
* Mean Squared Error (MSE): 0.5259818403639591
* Root Mean Squared Error (RMSE): 0.7252460550488773

### GLM

The GLM was the worst performing model, showing a huge deviance, Chi squared... Altogether it was just a fun try.


## Conclusions

The best performing model so far has been the NuSVR with the `rfb` (Radial Basis Function) kernel, with a MSE of 0.7071887919114409.

Although it is not a horrible result, there's obviously much room for improvement. Which is why we have recently started to implement BERT models.

## Future work

There's still more than a month for the competition to end, and so far we have testes what we could consider "traditional" models.

For the future of this project we are going to continue with BERT-models. [BERT](https://arxiv.org/abs/1810.04805?source=post_page) (Bidirectional Encoder Representations from Transformers) is a language representation model introduced in 2018. Since then, a family of BERT-like models like RoBERTa (Robustly Optimized BERT Pre-training Approach), ALBERT (A Little BERT)... We have started to fiddle with such models, and they will be the future models in which we'll work.
