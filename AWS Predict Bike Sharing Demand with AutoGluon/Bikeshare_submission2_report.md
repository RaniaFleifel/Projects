
# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Rania Tarek Fleifel


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Initial Training](#initial)
* [Exploratory data analysis and feature creation](#eda)
* [Hyperparameter tuning](#hpo)
* [Future work](#futurework)
* [Results](#results)
* [Summary](#summary)

## General info <a name="general-info"></a>
This project is the first of the AWS Machine learning engineer nano-degree. 
	
## Technologies <a name="technologies"></a>
    Project is created with AWS-Sagemaker
    - ml.t3.medium instance (2 vCPU + 4 GiB)
    - Notebook should be using kernal: Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)


## Initial Training <a name="initial"></a>
#### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
    I realized that kaggle does not accept any negative predictions. The scoring of kaggle for this competition depends on RMSlE between predicted and actual counts of passengers expected for each datetime. Since the actual counts are positive ints at all times, having the predicted as negative values would mess-up the evaluation of the proposed model.  

#### What was the top ranked model that performed?
 Weighted ensemble L3 

## Exploratory data analysis and feature creation <a name="eda"></a>
#### What did the exploratory analysis find and how did you add additional features?
Through EDA, it's clear that 
- *registered* movement has the highest correlation to *count*; this is obvious through hist and also the high correlation between the two variables.

- Although the count of rides is almost consistent throughout the 4 *seasons*, as the *weather* gets worse the count of bike riders decrease, even though the amount of data provided for the 4 seasons is uniform.

- The skewness of variable *holiday* is the highest, meaning that days that are not holidays *holiday=0* contribute more to the final count of riders. 

- It's best that *casual* and *registered* are excluded from the training because they're directly related to *count* variable and would over-ride the effect of the rest of the variables.

- *datetime* variable provide little insight as is.

#### How much better did your model preform after adding additional features and why do you think that is?
- **Drastically better than raw run**

    Improvement after adding new features= $100*abs(score{eda}-score{raw})/score{raw}=64.64$%
    
    
- All added features depdended initially on spreading *datetime* variable to *'day_of_week','hour_of_day','month'*
- All new features are added to both training and testing set, and Does not derive any information from test sets, naturally.
- The added features and resoning behind each:

1) *peak_hour_indicator*

My hypothesis was that depending on whether a day is a working day/a weekend/a holiday, the peak hours change. For instance, on working days typically between 9 to 5 have peaks around (8:00 am, 9:00 am and 5:00 pm), while weekends and holidays have lower traffic earlier in the day. so I sort the 24 hrs (ascendingly) in the following training cases seperately *working_day=1,working_day=0,holiday=1* depending on *training count* and create a new variable that is a number between 0:23 that represent how likely this hour would have high traffic, depending on which case of the aforementioned it fits.

2) *day_of_week_sin,day_of_week_cos,month_sin,month_cos*

Having days of the week and months of the year represented as numbers infer an issue that some of the models could deal with this data as ***cardinal***. For instance Monday:0 less priority than Sunday:6, similarily with January:0 and December:11. This data is **ordinal** and also **cyclic** 0 comes after 6 in days and also 0 comfes after 11 in months, a way to mitigate this was mentioned in <a href="https://stats.stackexchange.com/questions/245866/is-hour-of-day-a-categorical-variable" target="_blank">this thread</a>. by dissolving the variables to a sin and cos, this way the relation between subsequent values are preserved through closer values yet the cyclic nature is maintained as well/  

3) *AM/PM* 

An extra identifier to whether the hour_of_day is AM or PM 

##### The new variables infer data from all these variables *datetime ['day_of_week','hour_of_day','month'], working_day, holiday* and it also provides extra information that attempts to counter confusion from the cardinal look of some variables. 

##### I'm convinced that removing *'day_of_week','hour_of_day','month'* deduced from *datetime* can now be safely removed. However autogluon processes them anyway as long as *datetime* variable is considered (more on that in the comments section).

## Hyperparameter tuning <a name="hpo"></a>

 
#### `hyperparameter_tune_kwargs`: 
    [epochs=240,searcher='random',scheduler='local']

  
#### `hyperparameter`:
    [RF,GBM,KNN models]
    
    
#### How much better did your model preform after trying different hyper parameters?
- **better than eda_run**

    Improvement after optimizing hp $= 100*abs(score_{eda}-score_{hpo})/score_{eda}=+13.94%$%

#### Why did you use these hyper parameters and why do you think hpo performed this way?

* *hyperparameter_tune_kwargs* parameters control how the model approaches HPO through # of HPO 'epochs' or 'num_trials' to run. It also controls the parameters **searcher** and **scheduler** that is in-charge of the HPO search 

    - Instead of setting **num_trials**, I opt to remove the constrain on **time_limit** and let each model saturate (the more time-consuming option)

 hyperparameter_tune_kwargs 
        {
        'searcher':'random',
        'scheduler':'local',
        'epochs': 240
        }

* As for *hyperparameter*, through trying verbosity=4 on **eda_run** I found that **LightGBMXT_BAG_L1** and **KNeighborsUnif_BAG_L1** contributed the most to the top model **WeightedEnsemble_L3** , While **RandomForestMSE_BAG_L1** and **KNeighborsDist_BAG_L1** had the biggest weights in forming **WeightedEnsemble_L2**, so I decided to set the hyperparameters of **RF,GBM,KNN** base models.

    1)  RF {
        'criterion': 'squared_error',
        'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']},
        'max_depth': ag.space.Int(lower=5, upper=19),
        'max_features': ag.space.Categorical('sqrt', 'log2')
        }
        
* The regression nature of the problem suggests that searching through other criterion such as gini,logloss or entropy are more suited for classification problems.
* Setting the maximum_depth of the tree to the number of features makes sense, since the drivers of the decision are limited to that number.The lower limit is arbituary not too small such that the tree reaches *under_fiting* conclusions.
* Ideally, I'd specify max_features to 'None' and boot_strap to False such that the whole dataset is involved in each tree, but that would be too tedious and time-consuming. Hence, I specify both 'sqrt' and 'log2' options for the number of features looked for during splitting. 
    
    
    2)  GBM { 
        'learning_rate' : ag.space.Real(1e-3,1e-2,default=0.01,log=True),
        'num_leaves': ag.space.Int(33, 60,default=40),#31 < 2^max_depth
        'max_depth':6,
        'extra_trees': ag.space.Categorical(True,False), 
        'boosting':ag.space.Categorical('gbdt')
        }
    
* Setting num_leaves to 2^max_depth could cause overfitting. Similar to RF where i choose the minimum depth=5, I set minimum number of leaves to 2^5 + 1=33, and the maximum must be <2^6 for example 60.
* The smaller the learning_rate, the more information are read from between the lines. The default learning_rate is 0.1 and I wanted to make sure the learning_rate is prioritized over run_time so the range is between 1e-3 to 1e-2. To make sure the model considers a learning_rate that is small enough to capture details, yet big enough to prevent over_fitting, I choose 1e-2 to be the first rate attempted. 
* I use extra_trees interchangeably. This variable is set to False if I guarantee no over-fitting would occur. Else, it's better to train some GBM models with it on to also speed-up the training since a feature would be chosing based on one random threshold in this case. 
    
    3)  KNN {
        'algorithm':'auto',
        'n_neighbors':ag.space.Int(7,10,default=7), 
        'weights': ag.space.Categorical('uniform', 'distance')
        }
        
## Future work <a name="futurework"></a>
### If you were given more time with this dataset, where do you think you would spend more time?

**FEATURE ENGINEERING** 

- I believe that not considering *datetime* altogether in training the models would yield an improvement. The added features , from my point of view, enclosed alot of data that is deravtive of the rest of the variables including datetime. But including a ns-datetime type variable meant that autogluon processed the variable into numeric days,weeks,years and hours which i attempted to avoid with the added features. 

**HYPER-PARAMETER OPTIMIZATION**

- The biggest obstacle is how time-consuming and expensive any exhuastive hpo trial would be especially that I work on AWS-sagemaker:

    1- specify auto_stack=True instead of attempting to change num_bag_folds and num_bag_sets and remove timelimit variable to let models run its course
    
    2- set for RF: max_features:'None' and boot_strap:False
    
    3- attempt to add more models to hpo such that the base learners could merge to form stronger ensembles 

## Results <a name="results"></a>

### A table with the models ran, the hyperparameters modified, and the kaggle score.

| model | preset | timelimit | searcher | scheduler | epochs | score | marginal_score
| --- | --- | --- | --- | --- | --- | --- | --- |
**initial** | best_quality | 600 | - | - | - | 1.80493 | -64.64%
**add_features** | best_quality | 600 | - | - | - | 0.63824 | 0
**hpo_hypertuning** | best_quality | - | random | local | 250 | 0.54930 | +13.94%

### A line plot showing the top model score for the three training runs during the project.

![model_train_score.png](img/model_train_score.png)


### A line plot showing the top kaggle score for the three prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary <a name="summary"></a>

    I believe feature engineering took center-stage in my analysis. The 64.64% improvement when new features were added spoke volumes about the significance of data-wrangling and pre-processing of data fed into regression models. I believe feature-engineering could be taken to the next level if the plan was to tailor data to be easily interpreted by a certain model, however the goal of this project was to get more acquainted with Autogluon. Since auto-gluon attempts all sorts of models, this is where feature engineering stops and hyperparameters optimization begins.

    Auto-gluon on this particular problem performs good enough without mainpulating its parameters. However, I believe that focusing on specific hypyerparameters of the base-models could cause more drastic improvements. I focused on three base models RF, GBM and KNN. In order to balance the time-consuming nature of searching for good hyper-parameters of models, I limited my hpo trial to only these three models. This lead to approximately 14% improvement when compared to adding new features with no HPO.   





