
# Regression-Model-Comparison
A Comparison of Lasso, Ridge, Elastic Net and Randon Forest on Survey Data
___

##Background
This project was conducted as part of a Damining Class (STA-9890). It involves an investigation into survey data utilizing Lasso, Ridge, Elastic Net and Random Forest. Essentially this project seeks to ascertain which of the named models performs best on our data set when trying to predict height.

**Lasso Regression**: Lasso stands for Least Absolute Shrinkage and Selection Operator is a regression technique that allows for an easy way in selecting important features from our dataset because it supresses the coefficient of many features to zero. It heavily penalizes the absolute size of coefficients and based on your tuning parameter; you can control how many of your coefficients you want to be pushed to zero. 

**Ridge Regression**: Ridge penalizes the aquared size of coefficents and as such it leads to overal smaller coefficients as it pushes them closer to zero. Essentially, Ridge allows us to shrink the the coefficeints of features which sometimes allows us an easier time to see which features are most important.

**Elastic-Net Regression**:  Elastic-Net is a combination between Lasso and Ridge. It blends penalizing based on absolute size and squared size of coefficients.

**Random Forest**: Random forest is a decision tree classification technique. Essentially it posseses many individual trees that can easily produce their own class prediction, which is then used to determine the overall prediction of the model. 

## The Data
The [data](https://github.com/OjeWilliams/Regression-Model-Comparison/blob/master/Data/mydata.csv) for this project was based on a 2013 survey of statitics students of Slovakian nationality aged between 15-30 at FSEV UK. After pre-processing, which included removing unsuitable features such as categorical variables there was a total of 52 features(columns) and 1010 observations(rows). 
The response variable (y) was height, and the predictors(x) were all the other features.

## Steps
First ensure that the data file and the project code are in the same working directory.
1. Install and load the neccessary packages.
2. Read in the data file and carefully select your response variable and the predictors.
3. Set 80% of the data as the training dataset and the remaining as the test dataset.
4. Create appropriate variables to stores the residuals from each model.
5. Plot the R-Square values boxplot of each model for both test and train.
6. Plot 10-Fold Cross Validation Curves for Lasso Ridge and Elastic-Net
7. Time each model so that they can be compared for performance.

## Findings
**Top Music predictors:** Classical, Musicals, Rock, Metal, Alternative. <br />
**Top Movie predictors:** Comedy, Sci-Fi, Horror, Fantasy, War. <br />
**Other Top Predictors:** Weight, Interest in Cars, Passive Sports, Shopping, Science and Technology <br />
Ridge, Elastic-Net and Lasso performed similarly while Random forest had a longer runtime.



A full breakdown of how the above steps were completed can be seen in the project [code](https://github.com/OjeWilliams/Regression-Model-Comparison/blob/master/9890FinalProj.R) and the results were presented in the presentation [slides](https://github.com/OjeWilliams/Regression-Model-Comparison/blob/master/Proposal%20and%20Presentation/STAT%209890%20Regression%20Presentation.pptx).
