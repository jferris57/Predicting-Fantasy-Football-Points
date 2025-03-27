# Predicting Fantasy Football Points

## Research Question: Can we predict how many fantasy points an NFL player will score?
In the world of fantasy sports and sports betting, gaining an edge is everything. Whether you're competing in a fantasy league with friends for bragging rights (and maybe a bit of cash) or analyzing odds at a professional sportsbook, knowing how players will perform is crucial.

Sportsbooks rely on highly accurate models and projections to turn a profit, while bettors and fantasy players use data-driven insights to make informed decisions. My analysis aims to help fantasy football players and sports bettors navigate the unpredictable and chaotic football season with more confidence.

**Data Problem**: The goal of this project was to develop a machine learning model that can predict a player’s fantasy football points based on their past performance and other relevant statistics.

**Expected Results**: The expected result of this project is a well-tuned model that can predict PPR fantasy points with low error.

The data I will be using in this project is from Kaggle and can be found [here](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022).

Included in this data is various stats for players on a yearly basis. This includes stats like receptions, targets, receiving yards, rushing yards, yards after catch, etc.

The Jupyter Notebook containing my work can be found [here](https://github.com/jferris57/Predicting-Fantasy-Football-Points/blob/main/Predicting_Fantasy_Football_Points.ipynb).

## Methods and Approach

To achieve our goal, we collected and processed a dataset containing various player statistics from past seasons. Our approach included:

- Cleaning the data to remove unnecessary or highly correlated features that could distort predictions.

- Testing multiple machine learning models, including Ridge Regression, Lasso Regression, Support Vector Machines (SVR), Decision Trees, K-Nearest Neighbors (KNN), and a Voting Regressor, which combines multiple models for improved accuracy.

- Expanding the feature set using polynomial transformations to capture complex relationships between player statistics and fantasy points.

- Training and evaluating a dense neural network to see if deep learning techniques could enhance prediction accuracy.

Let's break down the process step by step:

## Understanding The Data
Initially looking at our data, there are a ton of features (195). There's all sorts of information like receiving yards, rush yards, injuries, interceptions, career yards after catch, height, weight, etc. I want to reduce this number and select a decent amount of useful features that are correlated with our target variable. The provided Jupyter Notebook lists every column in the dataset.

The dataset provides samples from seasons 2012-2023.

## Data Preparation
Here we will perform data cleaning to ensure our data is ready for modeling.

I removed the 'game type' feature as there was only one value. I identified our target as the 'fantasy points ppr' feature. I also checked the null count and converted this to a percentage of all the datapoints.

The 'vacated' columns consisted of a lot of missing data (6.33% of the dataset) and there was 1 value missing from the 'college' column.

I removed the one row containing missing college info and then checked the correlation between the 'vacated' features and our target. All of these features had low correlation ranging between -0.009 to -0.05 so I removed them entirely.

I then noticed there are some columns with weird values like 'inf' and '-inf' as well as some columns with very large numbers in the negatives. I decided to remove the data that had inf and -inf. 

Some of the columns containing large negative numbers were 'passer rating' and 'delta passer rating'. Passer ratings in the NFL are on a scale from 0 to 158.3. For this reason, I replaced any negative values with 0. As for the other columns with very negative values, I set them to values at the 5th percentile.

After this, the data was clean of missing values and very large negative values.

## Visualizations
Now that our data is clean, I wanted to investigate some of the data visually.

Here is the distribution of fantasy points over the whole dataset. This data is heavily skewed, but this is expected as we usually have few 'elite' fantasy players. I have highlighted where the mean and outliers lie.

![distribution](https://github.com/user-attachments/assets/603ebcd1-2ab1-436c-a394-0d7eda531b24)

I then looked at the fantasy points scored by each team:

![fantasy-points-by-team](https://github.com/user-attachments/assets/bfa2719a-e89d-4c09-b5f9-b48b793ea266)

The team points dont look much different from each other, we expect better teams to allow for more fantasy points than worse ones.

Here are the fantasy points scored by position:

![fantasy-points-by-position](https://github.com/user-attachments/assets/334de6a8-1aff-4013-9428-a0e51fd0824f)

The positional fantasy data was expected as usually QB's will have a high fantasy point output followed by WR's and RB's. It looks like running backs have a slightly lower mean than wide receivers and I believe this is expected because usually there are more viable wide receivers in fantasy than running backs.

I then looked at receiving yards by position, rushing yards by position, and total yards by position:

![receiving-yards-by-position](https://github.com/user-attachments/assets/46c9abde-0152-4387-93fa-b1254d8f1429)

![rushing-yards-by-position](https://github.com/user-attachments/assets/b766e1fe-fe43-46f6-b163-e2e7d6347b1d)

![total-yards](https://github.com/user-attachments/assets/9f3b43c4-bc00-4c7b-a6c0-3eaf2d10d950)

I then looked at points per game by position with QB's leading, followed by WR's then RB's and finally TE's.

![ppg](https://github.com/user-attachments/assets/05f70f03-ab77-41f8-81fd-909e9786b28f)

Finally, I looked at the fantasy point output trend for AJ Brown and CeeDee Lamb. At a very high level we can see that barring any injuries, we expect fantasy point output to trend upward as players become more experienced and play more seasons.

![brown](https://github.com/user-attachments/assets/fb6bde84-48b9-43c0-88c4-0f948f6a0ff0)

![lamb](https://github.com/user-attachments/assets/b0d031df-73bb-49ca-9d11-9e7da47fbef9)

## Engineering Features

The data has been cleaned but there is still an enormously large amount of features. Here we will choose the best features to be used in the modeling. 

In my initial Exploratory Data Analysis, I suspected a problem with multicollinearity because my baseline default linear regression model had an extremely low MSE. Here, I checked if the independent features were highly correlated with themselves. I created a correlation matrix and filtered by the feature pairs that had a correlation of 0.9 or greater. Just as I suspected, there were a lot. These can be found in the attached Jupyter notebook. Here are just a few to give you an example:

- Highly correlated features: complete_pass and pass_attempts

- Highly correlated features: incomplete_pass and pass_attempts

- Highly correlated features: incomplete_pass and complete_pass

- Highly correlated features: passing_yards and pass_attempts

- Highly correlated features: passing_yards and complete_pass

Right off the bat I removed a bunch of features that were extremely correlated with other independent variables but I also deemed not important for our project. Some of these include 'passing air yards', 'comp_pct', 'wins', 'losses'. At this step, I am trying to find the best features to keep and remove any redundant ones. I selected only the features with a correlation greater than 0.3 or less than -0.3 so we could use good features that were predictive of our target variable. The attached Jupyter notebook will list all of these correlation scores.

I then realized I did not want to keep player names, player ID's, or colleges. Player names or ID's are just identifiers and I dont want to One Hot Encode them and use them to train a model. As for colleges, there was a very large amount of them and since I don't think they very strongly can predict fantasy points, I removed those too. However, I did create a separate dataframe with these values to use later so that we can present our findings.

The last remaining categorical variables were team and position. There are 32 teams so I did not want to One Hot Encode these and create a large amount of new features. Instead, I analyzed their impact to fantasy point output by grouping our dataset by team and taking the mean fantasy point output. This resulted in a numerical value that could determine which teams created more fantasy point output on average than others. I created a new column of these values and removed the 'team' column. 

The last categorical feature is position, and I will One Hot Encode these values for model training.

Some of the remaining features in the dataset were redundant and needed to be removed. For example, I removed 'draft round' and 'draft pick' because we already have overall draft position. I removed 'total yards' and 'total tds' because these are just sums of the different types of yards or touchdowns. Pass attempts and rush attempts had an extremely strong correlation with other stats, as did targets. 'Passing yards' implies there were completed passes so 'completed passes' could be removed. This was probably the hardest part of the project as I needed to decide which features were most important and how much each one was negatively impacting the models.

## Modeling
### Baseline Model
For the baseline model, I used a Linear Regression model with default parameters. The results were much better than in my initial data analysis as it seems the model is not grossly overfit this time.

![image](https://github.com/user-attachments/assets/2cfbf0f3-3811-4234-af58-3f31cde60e4f)

### Tuned Individual Models
Next, I trained Ridge, Lasso, SVR, KNeighborsRegressor, and DecisionTreeRegressor models with hyperparameter tuning. Here are the results:

![image](https://github.com/user-attachments/assets/5ab08ebb-c7b8-46db-b0a3-59260b5abed6)

Here we can see that the Decision Tree and KNeighbors models are not performing well at all. Ridge, Lasso, and SVR all performed well and will be included in the ensemble model.

### Ensemble Model
Next, I trained an ensemble model, which combines machine learning models in hopes to get even better results. I chose to combine the Ridge, Lasso, and SVR models. Here are the results:

![image](https://github.com/user-attachments/assets/08be2406-265b-4e2e-ba9c-ebbc221cdef8)

This model has a slight improvement in both R2 score and Test MSE. So far this is the best model we've trained.

### Polynomial Features
I wanted to take this a step further and see if capturing non-linear relationships between stats could help our models. I transformed the data and re-trained Ridge, Lasso, and SVR models to find the most optimized parameters. Here are the results:

![image](https://github.com/user-attachments/assets/020df2d5-363f-4d57-a45b-47aa65b4a9cb)

Here we can see a massive increase in performance. R2 scores improved and Test MSE drastically improved. This means these models are better at explaining the variation in fantasy football points and can predict points more accurately!

I combined these three models into another ensemble model:

![image](https://github.com/user-attachments/assets/ec7ced13-676c-4bad-ab0f-8eb88788de14)

The results improved slightly again from the individual models. This is now the best model we have for predicting fantasy points.

### Neural Network
I wanted to try training a neural network on this data and see how it performed against traditional machine learning models. The model did pretty well, but it requires more tuning.

![image](https://github.com/user-attachments/assets/feab80e0-93a0-4ec8-9215-8f529de53256)

## Key Findings
Let's sum up what we've found:

1. Model Performance:
    - The best performing model was an Ensemble Model combining multiple other models and used polynomial data
    - Polynomial transformations greatly improved performance, showing us that non-linear relationships between player stats and fantasy points help the models
    - The model can predict fantasy points with around 4-5 points of error
2. Important Features:
    - The most important features in predicting fantasy points were receiving yards, rushing yards, passing yards, touchdowns, interceptions, draft pick, yards after catch, offense snaps, passer rating just to name a few. More can be found in the attached notebook.
    - Features that were highly correlated with each other such as total yards, total tds, pass attempts, targets, draft round were removed to avoid redundancy and help the models
3. Challenges
    - The Decision Tree and KNN models severly overfitted the training data and failed to perform well for this problem
    - The SVR model takes a significant amount of time to train

## Next Steps and Recommendations
  - **Testing future data**: This model has performed well on the test data, but it should be tested with future game data like the 2024 season
  - **Incorporate weather, defense, and injury**: Adding player injuries, opponent defense stats, and weather could further help the model with accuracy
  - **Develop a user-friendly UI**: This model should be implemented in a website or app to help everyday fantasy football players use it easily
  - **Continuous updates**: This model should be updated with new and current data whenever possible. The 2024 season data should be added to it so it can train on it and predict 2025 data

## Conclusion
This project demonstrated that machine learning can be a powerful tool in the world of fantasy sports and can be used to accurately predict fantasy points. By cleaning the data, selecting the right features, and optimizing models, we are left with a highly accurate model for fantasy football projections. As this model gets updated with new player data and new features to train on, this will become an extremely useful tool for fantasy football enthusiasts and sports bettors.
