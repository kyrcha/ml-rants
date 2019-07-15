# ml-rants

Various machine learning code and pipelines, in various languages. Used also to support some of my blog posts at: [kyrcha.info](http://kyrcha.info).

## Contents

 - [Collinearity and feature selection](https://github.com/kyrcha/ml-rants/blob/master/CollinearityAndFeatureSelection.md)
 - [Optimizing XGBoost and Random Forests with Bayesian Optimization](https://github.com/kyrcha/ml-rants/blob/master/xgboost_rbf_bayesian_opt.ipynb)

## Rendering Instructions

Rendering instructions for Rmd documents.

To render Rmd in markdown:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'md_document')

To render Rmd in html:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'html_document', output_dir = './docs')

