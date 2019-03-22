Rmd document for Collinearity and Feature Selection
===================================================

Intro
-----

This notebook is an online appendix of my blog post: [On Collinearity
and Feature
Selection](http://kyrcha.info/2019/03/22/on-collinearity-and-feature-selection),
where I play with the concepts using R code.

The dataset
-----------

We will use the [auto-mpg
dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg), where we
will try to predict the miles per galon (mpg) consumption given some car
related features like horsepower, weight etc.

    set.seed(1234)
    fileURL <- "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    download.file(fileURL, destfile="auto-mpg.data", method="curl")
    data <- read.table("auto-mpg.data", na.strings = "?", quote='"', dec=".", header=F)
    # remove instances with missing values and the name of the car
    data <- data[complete.cases(data),-9]
    summary(data)

    ##        V1              V2              V3              V4       
    ##  Min.   : 9.00   Min.   :3.000   Min.   : 68.0   Min.   : 46.0  
    ##  1st Qu.:17.00   1st Qu.:4.000   1st Qu.:105.0   1st Qu.: 75.0  
    ##  Median :22.75   Median :4.000   Median :151.0   Median : 93.5  
    ##  Mean   :23.45   Mean   :5.472   Mean   :194.4   Mean   :104.5  
    ##  3rd Qu.:29.00   3rd Qu.:8.000   3rd Qu.:275.8   3rd Qu.:126.0  
    ##  Max.   :46.60   Max.   :8.000   Max.   :455.0   Max.   :230.0  
    ##        V5             V6              V7              V8       
    ##  Min.   :1613   Min.   : 8.00   Min.   :70.00   Min.   :1.000  
    ##  1st Qu.:2225   1st Qu.:13.78   1st Qu.:73.00   1st Qu.:1.000  
    ##  Median :2804   Median :15.50   Median :76.00   Median :1.000  
    ##  Mean   :2978   Mean   :15.54   Mean   :75.98   Mean   :1.577  
    ##  3rd Qu.:3615   3rd Qu.:17.02   3rd Qu.:79.00   3rd Qu.:2.000  
    ##  Max.   :5140   Max.   :24.80   Max.   :82.00   Max.   :3.000

Preprocessing
-------------

Let's normalize the dataset (values to be in the interval \[0,1\]), an
operation that will maintain the correllation between the variables, and
split between training and testing.

    normalize <- function(x) {
      (x - min(x, na.rm=TRUE))/(max(x,na.rm=TRUE) - min(x, na.rm=TRUE))
    }
    normData <- cbind(data[,1], as.data.frame(lapply(data[,-1], normalize)))

    # name variables
    names(normData) <- c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin")

    # check correlation
    cat("Cor between disp. and weight before norm.:", cor(data$V3, data$V5), "\n")

    ## Cor between disp. and weight before norm.: 0.9329944

    cat("After norm.:", cor(normData$displacement, normData$weight))

    ## After norm.: 0.9329944

    # Train/Test split
    library(tidyverse)
    library(caret)
    training.samples <- normData$mpg %>%
    createDataPartition(p = 0.8, list = FALSE)
    train.data  <- normData[training.samples, ]
    test.data <- normData[-training.samples, ]

Modelling
---------

### Linear Regression

    linearModel <- lm(mpg ~., data = train.data)
    summary(linearModel)

    ## 
    ## Call:
    ## lm(formula = mpg ~ ., data = train.data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.4147 -2.1845 -0.1875  1.7702 12.8931 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   25.4017     1.3012  19.521  < 2e-16 ***
    ## cylinders     -2.5022     1.8711  -1.337   0.1821    
    ## displacement   7.7778     3.2280   2.409   0.0166 *  
    ## horsepower    -2.8404     2.8077  -1.012   0.3125    
    ## weight       -22.9293     2.5394  -9.029  < 2e-16 ***
    ## acceleration   2.0678     1.8409   1.123   0.2622    
    ## model_year     9.3472     0.7023  13.309  < 2e-16 ***
    ## origin         2.9604     0.6383   4.638 5.21e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.389 on 307 degrees of freedom
    ## Multiple R-squared:  0.8175, Adjusted R-squared:  0.8134 
    ## F-statistic: 196.5 on 7 and 307 DF,  p-value: < 2.2e-16

One can see that weight is a very important factor both in terms of the
coefficient value and in terms of statistical significance (unlikely to
observe a relationship between weight and mpg due to change). Notice the
negative coefficient (more weight, less miles per gallon), which can be
explained by the laws of physics. But also notice that even though
weight and diplacement have a correlation o 0.93 (almost collinear),
their coefficients have different signs. Based on common knowledge
though they should have had the same signs. Collinearity is bad when you
try and explain the outputs of models. Let's examine the VIF values:

    library(car)
    vif(linearModel)

    ##    cylinders displacement   horsepower       weight acceleration 
    ##    10.797995    19.995685     8.952301    10.191612     2.424953 
    ##   model_year       origin 
    ##     1.223735     1.794710

We observe that 3 predictors have a value more than 10 that is a concern
for the existence of collinearity (or multicollinearity in this case).
So let's drop displacement and create a second model:

    linearModelMinusDisp <- lm(mpg ~.-displacement, data = train.data)
    summary(linearModelMinusDisp)

    ## 
    ## Call:
    ## lm(formula = mpg ~ . - displacement, data = train.data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.4776 -2.2073 -0.1473  1.7625 12.9679 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   25.4241     1.3113  19.388  < 2e-16 ***
    ## cylinders      0.5072     1.4040   0.361    0.718    
    ## horsepower    -1.1352     2.7382  -0.415    0.679    
    ## weight       -20.6134     2.3688  -8.702  < 2e-16 ***
    ## acceleration   1.5673     1.8434   0.850    0.396    
    ## model_year     9.2684     0.7070  13.110  < 2e-16 ***
    ## origin         2.4812     0.6112   4.060 6.24e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.415 on 308 degrees of freedom
    ## Multiple R-squared:  0.8141, Adjusted R-squared:  0.8104 
    ## F-statistic: 224.7 on 6 and 308 DF,  p-value: < 2.2e-16

    vif(linearModelMinusDisp)

    ##    cylinders   horsepower       weight acceleration   model_year 
    ##     5.986398     8.383554     8.731646     2.394082     1.221086 
    ##       origin 
    ##     1.620491

and let's check the predictive ability of the two:

    predLM <- linearModel %>% predict(test.data)
    predLMMD <- linearModelMinusDisp %>% predict(test.data)

    cat("Full model:", RMSE(predLM, test.data$mpg), "\n")

    ## Full model: 3.098026

    cat("Minus disp. model:", RMSE(predLMMD, test.data$mpg))

    ## Minus disp. model: 3.125825

As one can see I now have a more "understandable" model, with kind of a
"worse" predictive ability (slightly higher error).

Now let's remove all the variables that had a VIF value of more than 10

    linearModelSimpler <- lm(mpg ~.-displacement-cylinders-weight, data = train.data)
    summary(linearModelSimpler)

    ## 
    ## Call:
    ## lm(formula = mpg ~ . - displacement - cylinders - weight, data = train.data)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -9.832 -2.415 -0.533  1.930 12.794 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   28.7618     1.4362  20.027  < 2e-16 ***
    ## horsepower   -24.3357     1.6960 -14.349  < 2e-16 ***
    ## acceleration  -6.9758     1.8700  -3.730 0.000227 ***
    ## model_year     8.0722     0.8034  10.048  < 2e-16 ***
    ## origin         4.9410     0.6324   7.813 8.76e-14 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.938 on 310 degrees of freedom
    ## Multiple R-squared:  0.7511, Adjusted R-squared:  0.7479 
    ## F-statistic: 233.9 on 4 and 310 DF,  p-value: < 2.2e-16

    vif(linearModelSimpler)

    ##   horsepower acceleration   model_year       origin 
    ##     2.418236     1.852449     1.185461     1.304459

    predLMS <- linearModelSimpler %>% predict(test.data)
    cat("Even simpler model - RMSE:", RMSE(predLMS, test.data$mpg), "\n")

    ## Even simpler model - RMSE: 3.575336

Now the model is simple, more explainable, without any colinearities
(low VIF values) but not as good in terms of RMSE as the previous ones.

Feature Selection
-----------------

To check also a feature selection method, stepwise feature selection
using the Akaike Information Criterion (AIC):

    require(leaps)
    require(MASS)
    step.model <- stepAIC(linearModel, direction = "both", trace = FALSE)
    summary(step.model)

    ## 
    ## Call:
    ## lm(formula = mpg ~ displacement + weight + acceleration + model_year + 
    ##     origin, data = train.data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.1253 -2.1573 -0.1547  1.8907 12.8220 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   24.4535     1.0334  23.663  < 2e-16 ***
    ## displacement   4.3108     2.3178   1.860   0.0639 .  
    ## weight       -24.4212     2.2367 -10.918  < 2e-16 ***
    ## acceleration   3.1711     1.4748   2.150   0.0323 *  
    ## model_year     9.5092     0.6811  13.963  < 2e-16 ***
    ## origin         2.7723     0.6225   4.453 1.18e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.392 on 309 degrees of freedom
    ## Multiple R-squared:  0.816,  Adjusted R-squared:  0.813 
    ## F-statistic:   274 on 5 and 309 DF,  p-value: < 2.2e-16

    linearModelAIC <- lm(as.formula(step.model), data = train.data)
    vif(linearModelAIC)

    ## displacement       weight acceleration   model_year       origin 
    ##    10.288515     7.890980     1.553344     1.148540     1.703980

    predLMAIC <- linearModelAIC %>% predict(test.data)
    cat("AIC model:", RMSE(predLMAIC, test.data$mpg), "\n")

    ## AIC model: 3.114763

Through this example we can see than even though we have colinearities
involved (VIF value of 10+), we obtain a low RMSE of 3.11. Or using
another feature selection package:

    # Set up repeated k-fold cross-validation
    train.control <- trainControl(method = "cv", number = 10)
    # Train the model
    step.model2 <- train(mpg ~., data = train.data,
                        method = "leapBackward", 
                        tuneGrid = data.frame(nvmax = 1:7),
                        trControl = train.control
                        )
    step.model2$results

    ##   nvmax     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1     1 4.387909 0.6950849 3.378332 0.9854225 0.09849010 0.6697571
    ## 2     2 3.407376 0.8137023 2.661535 0.9123314 0.06485389 0.5518240
    ## 3     3 3.333900 0.8224665 2.549990 0.8779426 0.06157152 0.5311275
    ## 4     4 3.371991 0.8188230 2.592064 0.8832272 0.06214344 0.5410169
    ## 5     5 3.395429 0.8170493 2.618763 0.8325787 0.05722386 0.4970754
    ## 6     6 3.404866 0.8163436 2.633234 0.7979961 0.05357786 0.4526265
    ## 7     7 3.384003 0.8180447 2.618266 0.8100225 0.05401179 0.4619242

    summary(step.model2$finalModel)

    ## Subset selection object
    ## 7 Variables  (and intercept)
    ##              Forced in Forced out
    ## cylinders        FALSE      FALSE
    ## displacement     FALSE      FALSE
    ## horsepower       FALSE      FALSE
    ## weight           FALSE      FALSE
    ## acceleration     FALSE      FALSE
    ## model_year       FALSE      FALSE
    ## origin           FALSE      FALSE
    ## 1 subsets of each size up to 3
    ## Selection Algorithm: backward
    ##          cylinders displacement horsepower weight acceleration model_year
    ## 1  ( 1 ) " "       " "          " "        "*"    " "          " "       
    ## 2  ( 1 ) " "       " "          " "        "*"    " "          "*"       
    ## 3  ( 1 ) " "       " "          " "        "*"    " "          "*"       
    ##          origin
    ## 1  ( 1 ) " "   
    ## 2  ( 1 ) " "   
    ## 3  ( 1 ) "*"

    coef(step.model2$finalModel, 3)

    ## (Intercept)      weight  model_year      origin 
    ##   26.190907  -21.244932    9.522434    2.370324

The best model has 3 predictors: `weight`, `model_year` and `origin`. So
making one more final linear regression model and predicting the `mpg`
in the test set we have:

    linearModelBest <- lm(mpg ~weight+model_year+origin, data = train.data)
    summary(linearModelBest)

    ## 
    ## Call:
    ## lm(formula = mpg ~ weight + model_year + origin, data = train.data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -9.8638 -2.1894 -0.0388  1.7413 13.0971 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  26.1909     0.6744  38.834  < 2e-16 ***
    ## weight      -21.2449     1.0134 -20.965  < 2e-16 ***
    ## model_year    9.5224     0.6649  14.321  < 2e-16 ***
    ## origin        2.3703     0.5942   3.989 8.28e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3.412 on 311 degrees of freedom
    ## Multiple R-squared:  0.8126, Adjusted R-squared:  0.8108 
    ## F-statistic: 449.6 on 3 and 311 DF,  p-value: < 2.2e-16

    vif(linearModelBest)

    ##     weight model_year     origin 
    ##   1.601095   1.082269   1.534712

    predLMBest <- linearModelBest %>% predict(test.data)
    cat("Best model:", RMSE(predLMBest, test.data$mpg), "\n")

    ## Best model: 3.096694

*3.09*!!! Lowest error, with only 3 predictors and no collinearities
involved. In this case feature selection went along with a fina model
that is interpretable as well.

Ridge Regression
----------------

One solution to the collinearity problem (without doing any feature
selection) is to apply Ridge Regression and to try to "constrain" the
number of solution of the `beta` coefficients into a single solution. In
this case since we have the hyperparameter lambda to optimize for which
we will apply 10-fold cross-validation on the training set to find the
best value and then use that to train the model and predict mpg in the
testing dataset.

    library(glmnet)
    y <- train.data$mpg
    x <- train.data %>% dplyr::select(-starts_with("mpg")) %>% data.matrix()
    lambdas <- 10^seq(3, -2, by = -.1)
    fit <- glmnet(x, y, alpha = 0, lambda = lambdas)
    cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas, nfolds = 10)
    # uncomment the plot to see how lambda changes the error.
    #plot(cv_fit)
    opt_lambda <- cv_fit$lambda.min
    x_test <- test.data %>% dplyr::select(-starts_with("mpg")) %>% data.matrix()
    y_predicted <- predict(fit, s = opt_lambda, newx = x_test)
    cat("Ridge RMSE:", RMSE(y_predicted, test.data$mpg))

    ## Ridge RMSE: 3.096991

The Ridge Regression produced one of the lowest error and without
dropping any of the coefficients. And as for the coefficients' values:

    coef(cv_fit)

    ## 8 x 1 sparse Matrix of class "dgCMatrix"
    ##                        1
    ## (Intercept)   26.7194742
    ## cylinders     -2.1671592
    ## displacement  -1.6690043
    ## horsepower    -5.5116891
    ## weight       -11.7117393
    ## acceleration  -0.3212087
    ## model_year     7.8818066
    ## origin         2.6205643

which as we can see gave a much more reasonable and physically
explainable model. The more cyclinders, displacement, horsepower, weight
and accellaration...the less miles per gallon you can drive, while the
more recent the model, which originated from origin 2 (Europe) or 3
(Japan) the more miles on the gallon you can go. As for the RMSE, it is
close both to the full model and to the optimized model using feature
selection.

Discussion
----------

Collinearity is important if you need to have an understandable model.
If you don't, and you just care for predictive ability you can be more
brute and care about the numbers.
