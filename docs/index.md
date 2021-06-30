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

`
'When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.\nThe floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.\nAt each end of the room, on the wall, hung a beautiful bear-skin rug.\nThese rugs were for prizes, one for the girls and one for the boys. And this was the game.\nThe girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.\nThis would have been an easy matter, but each traveller was obliged to wear snowshoes.'
`

The `target` is the reading difficulty. It is the result of a [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley-Terry_model) analysis of more than 111,000 pairwise comparisons between excerpts. Teachers spanning grades 3-12 (a majority teaching between grades 6−10) served as the raters for these comparisons.

The `standard_error` is included as an output of the Bradley-Terry analysis because individual raters saw only a fraction of the excerpts, while every excerpt was seen by numerous raters. 

On average, each excerpt has been reviewed 22 times.

## Text treatment

The data treatment we have carried out involves the following steps:

* Make the words lower case.
* Remove stop words: 
Words like tal
* Tokenize words: do this to words: example
* Remove punctuation
* Stemmatize

After going through this process, an expert such as the one shown before looks like this:

`
'young peopl return ballroom present decid chang appear instead interior scene winter landscap floor cover snowwhit canva laid smooth rumpl bump hillock like real snow field numer palm evergreen decor room powder flour strewn tuft cotton like snow also diamond dust light sprinkl them glitter crystal icicl hung branch end room wall hung beauti bearskin rug rug prize one girl one boy game girl gather one end room boy other one end call north pole south pole player given small flag plant reach pole would easi matter travel oblig wear snowsho'
`


## Text measurement

Statistics, complexity measures and so on

## Model evaluation

## Conclusions

## Future work



Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/JuanCoRo/CommonLit-Readability/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
