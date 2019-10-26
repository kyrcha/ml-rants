# ml-rants

Various machine learning code and pipelines, in various languages. Used also to support some of my blog posts at: [kyrcha.info](http://kyrcha.info). Since GitHub does not render Rmd notebooks, these notebooks are rendered using [Rpubs](http://rpubs.com/kyrcha).

## Contents

 - [Collinearity and feature selection](https://github.com/kyrcha/ml-rants/blob/master/CollinearityAndFeatureSelection.md)
 - [Optimizing XGBoost and Random Forests with Bayesian Optimization](https://github.com/kyrcha/ml-rants/blob/master/xgboost_rbf_bayesian_opt.ipynb)
 - [Fitting modified Gompertz and Baranyi equations for bacterial growth in R](https://github.com/kyrcha/ml-rants/blob/master/gompertz-baranyi-example.Rmd) - [Rendered](http://rpubs.com/kyrcha/gompertz-baranyi-fit)

## Rendering Instructions

Rendering instructions for Rmd documents.

To render Rmd in markdown:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'md_document')

To render Rmd in html:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'html_document', output_dir = './docs')

