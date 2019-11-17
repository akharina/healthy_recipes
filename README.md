# healthy_recipes
A statistical analysis and classification exercise on Epicurious' 'healthy' recipes

## Data source
The dataset used in this repo was downloaded from ["Epicurious - Recipes with Rating and Nutrition" on Kaggle](https://www.kaggle.com/hugodarwood/epirecipes). 

## Notes and findings

### Statistical testing 
- Bootstrap t-test shows that there is no significant difference between the average nutritional values (fat, protein, sodium, calories) of recipes tagged "healthy" and untagged recipes.

### Classification
- Since we have highly imbalanced classes between "healthy" and untagged recipes, resampling methods are used.
- Resampling the train data was found to be prone to overfitting compared to performing k-fold stratification and resampling via cross-validation within pipelines. 
- The classification model performance (~70% accuracy using XGBoost) is not very good given the highly imbalanced data, suggesting that there are many healthy recipes that are untagged. 

## Blog post 
A less-technical [blog post](https://medium.com/@anastasia.kharina/a-statistical-analysis-and-classification-exercise-on-epicurious-healthy-recipes-d2d09eb2dee1) is published on [Medium](https://medium.com/) to accompany this repo. 
