# ML rants

Various code, in various language for machine learning stuff. Used also to support experimentation of my website and blog: [kyrcha.info](http://kyrcha.info).

## Instructions

To render Rmd in markdown:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'md_document')

To render Rmd in html:

    rmarkdown::render('CollinearityAndFeatureSelection.Rmd', output_format = 'html_document', output_dir = './docs')