# Machine Learning Performance



# Regression Model Performance

   Model | 5-Fold CV Avg R2 Score | 5-Fold CV Avg RMSE (normalized) | Avg RMSE (in $ value) | Avg R2 Score | Relative Error | RMSE (normalized) 
|:------|:------:| :------: |:------:|:------:|:------:|:------:|
|Linear Regression|30.946| 40.004| 8085.094 |31.04| 31.94 | 40.05|
|Linear SVR | 27.36| 41.03| 8292.456| 27.4|31.24|41.09|
|SGD Regressor | 30.92| 40.008| 8085.903|31.03|31.94|40.05|
|Gradient Boosting Regressor |48.992|34.376| 6947.635|48.47| 26.81| 34.61|
|Decision Tree Regressor|83.116|19.78 &| 3997.679|83.49|13.02|19.59|
|Random Forest Regressor| 90.686|14.688| 2968.55|90.78|9.84|14.64|
| Extra Trees Regressor |90.06|15.178 | 3067.582|90.18|10.14|15.11|
| Bagging Regressor |89.934|15.272| 3086.58|90.06|10.24|15.2|
| Gradient Boosting Regressor|66.774| 27.746 | 5607.665|66.97| 20.71| 27.71|
| Ada Boost Regressor |73.442| 24.786| 5009.428| 73.84| 20.57| 24.66|


# Classification Model Performance

|	 Model 	| 5-Fold CV Avg Accuracy | Avg Accuracy |
|	:------ |		:------:		 |	 :------:	|
| Nearest Neighbors Classifier | 37.684 | 38.31 |
| Decision Tree Classifier | 47.578 | 48.18 |
| Random Forest Classifier | 53.276 | 53.87 |
| Extra Trees Classifier | 52.976 | 53.62 |
| Neural Net Classifier | 23.952 | 23.96 |
|  AdaBoost Classifier | | |


# Starting 5 Fold Cross Validation for Regressors.

## Training for Linear Regression starting ****

### training performance
acc(r2_score) for training = 31.02  
acc(relative error) for training = 31.96  
acc(rmse) for training = 39.97  
### Test performance
acc(r2_score) for training = 30.65  
acc(relative error) for training = 32.13  
acc(rmse) for training = 40.13  

### training performance
acc(r2_score) for training = 30.92  
acc(relative error) for training = 32.02  
acc(rmse) for training = 40.04  
### Test performance
acc(r2_score) for training = 31.07  
acc(relative error) for training = 31.85  
acc(rmse) for training = 39.86  

### training performance
acc(r2_score) for training = 31.0  
acc(relative error) for training = 32.0  
acc(rmse) for training = 40.0  
### Test performance
acc(r2_score) for training = 30.72  
acc(relative error) for training = 31.95  
acc(rmse) for training = 40.0  

### training performance
acc(r2_score) for training = 30.92  
acc(relative error) for training = 31.99  
acc(rmse) for training = 40.0  
### Test performance
acc(r2_score) for training = 31.05  
acc(relative error) for training = 32.0  
acc(rmse) for training = 40.03  

### training performance
acc(r2_score) for training = 30.87  
acc(relative error) for training = 31.99  
acc(rmse) for training = 40.0  
### Test performance
acc(r2_score) for training = 31.24  
acc(relative error) for training = 32.01  
acc(rmse) for training = 40.0  

Avg R2 Score: 30.946  
Avg RMSE (normalized): 40.004 & in $ value: 8085.094  

### Test performance before pickling
acc(r2_score) for training = 31.04  
acc(relative error) for training = 31.94  
acc(rmse) for training = 40.05  
### Test performance after pickling
acc(r2_score) for training = 31.04  
acc(relative error) for training = 31.94  
acc(rmse) for training = 40.05  


## Training for Linear SVR starting ****

### training performance
acc(r2_score) for training = 27.41  
acc(relative error) for training = 31.24  
acc(rmse) for training = 41.0  
### Test performance
acc(r2_score) for training = 27.14  
acc(relative error) for training = 31.38  
acc(rmse) for training = 41.14  

### training performance
acc(r2_score) for training = 27.27  
acc(relative error) for training = 31.29  
acc(rmse) for training = 41.08  
### Test performance
acc(r2_score) for training = 27.36  
acc(relative error) for training = 31.18  
acc(rmse) for training = 40.92  

### training performance
acc(r2_score) for training = 27.45  
acc(relative error) for training = 31.28  
acc(rmse) for training = 41.02  
### Test performance
acc(r2_score) for training = 27.14  
acc(relative error) for training = 31.24  
acc(rmse) for training = 41.02  


### training performance
acc(r2_score) for training = 27.33  
acc(relative error) for training = 31.27  
acc(rmse) for training = 41.02  
### Test performance
acc(r2_score) for training = 27.41  
acc(relative error) for training = 31.28  
acc(rmse) for training = 41.07  


### training performance
acc(r2_score) for training = 27.36  
acc(relative error) for training = 31.27  
acc(rmse) for training = 41.01  
### Test performance
acc(r2_score) for training = 27.75  
acc(relative error) for training = 31.28  
acc(rmse) for training = 41.0  

Avg R2 Score: 27.36  
Avg RMSE (normalized): 41.03 & in $ value: 8292.456  

### Test performance before pickling
acc(r2_score) for training = 27.4  
acc(relative error) for training = 31.24  
acc(rmse) for training = 41.09  
### Test performance after pickling
acc(r2_score) for training = 27.4  
acc(relative error) for training = 31.24  
acc(rmse) for training = 41.09  

## Training for SGD Regressor starting ****

### training performance
acc(r2_score) for training = 30.99  
acc(relative error) for training = 32.01  
acc(rmse) for training = 39.98  
### Test performance
acc(r2_score) for training = 30.62  
acc(relative error) for training = 32.19  
acc(rmse) for training = 40.14  

### training performance
acc(r2_score) for training = 30.9  
acc(relative error) for training = 31.99  
acc(rmse) for training = 40.04  
### Test performance
acc(r2_score) for training = 31.05  
acc(relative error) for training = 31.83  
acc(rmse) for training = 39.86  

### training performance
acc(r2_score) for training = 31.0  
acc(relative error) for training = 32.03  
acc(rmse) for training = 40.01  
### Test performance
acc(r2_score) for training = 30.71  
acc(relative error) for training = 31.98  
acc(rmse) for training = 40.0  

### training performance
acc(r2_score) for training = 30.9  
acc(relative error) for training = 32.08  
acc(rmse) for training = 40.0  
### Test performance
acc(r2_score) for training = 31.03  
acc(relative error) for training = 32.1  
acc(rmse) for training = 40.03  

### training performance
acc(r2_score) for training = 30.82  
acc(relative error) for training = 31.94  
acc(rmse) for training = 40.02  
### Test performance
acc(r2_score) for training = 31.19  
acc(relative error) for training = 31.96  
acc(rmse) for training = 40.01  

Avg R2 Score: 30.92  
Avg RMSE (normalized): 40.008 & in $ value: 8085.903  

### Test performance before pickling
acc(r2_score) for training = 31.03  
acc(relative error) for training = 31.94  
acc(rmse) for training = 40.05  
### Test performance after pickling
acc(r2_score) for training = 31.03  
acc(relative error) for training = 31.94  
acc(rmse) for training = 40.05  

## Training for Gradient Boosting Regressor starting ****

### training performance
acc(r2_score) for training = 47.17  
acc(relative error) for training = 27.26  
acc(rmse) for training = 34.98  
### Test performance
acc(r2_score) for training = 46.88  
acc(relative error) for training = 27.4  
acc(rmse) for training = 35.13  

### training performance
acc(r2_score) for training = 47.0  
acc(relative error) for training = 27.27  
acc(rmse) for training = 35.07  
### Test performance
acc(r2_score) for training = 46.9  
acc(relative error) for training = 27.19  
acc(rmse) for training = 34.98  

### training performance
acc(r2_score) for training = 49.59  
acc(relative error) for training = 26.56  
acc(rmse) for training = 34.19  
### Test performance
acc(r2_score) for training = 49.22  
acc(relative error) for training = 26.6  
acc(rmse) for training = 34.25  

### training performance
acc(r2_score) for training = 50.19  
acc(relative error) for training = 26.3  
acc(rmse) for training = 33.96  
### Test performance
acc(r2_score) for training = 50.34  
acc(relative error) for training = 26.31  
acc(rmse) for training = 33.97  

### training performance
acc(r2_score) for training = 51.35  
acc(relative error) for training = 25.91  
acc(rmse) for training = 33.56  
### Test performance
acc(r2_score) for training = 51.62  
acc(relative error) for training = 25.9  
acc(rmse) for training = 33.55  

Avg R2 Score: 48.992  
Avg RMSE (normalized): 34.376 & in $ value: 6947.635  

### Test performance before pickling
acc(r2_score) for training = 48.47  
acc(relative error) for training = 26.81  
acc(rmse) for training = 34.61  
### Test performance after pickling
acc(r2_score) for training = 48.47  
acc(relative error) for training = 26.81  
acc(rmse) for training = 34.61  

## Training for Decision Tree Regressor starting ****

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.41  
### Test performance
acc(r2_score) for training = 83.41  
acc(relative error) for training = 13.13  
acc(rmse) for training = 19.63  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 83.04  
acc(relative error) for training = 13.14  
acc(rmse) for training = 19.77  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 82.9  
acc(relative error) for training = 13.22  
acc(rmse) for training = 19.87  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 83.15  
acc(relative error) for training = 13.17  
acc(rmse) for training = 19.79  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.41  
### Test performance
acc(r2_score) for training = 83.08  
acc(relative error) for training = 13.18  
acc(rmse) for training = 19.84  

Avg R2 Score: 83.116  
Avg RMSE (normalized): 19.78 & in $ value: 3997.679  

### Test performance before pickling
acc(r2_score) for training = 83.49  
acc(relative error) for training = 13.02  
acc(rmse) for training = 19.59  
### Test performance after pickling
acc(r2_score) for training = 83.49  
acc(relative error) for training = 13.02  
acc(rmse) for training = 19.59  

## Training for Random Forest Regressor starting ****

### training performance
acc(r2_score) for training = 98.69  
acc(relative error) for training = 3.69  
acc(rmse) for training = 5.5  
### Test performance
acc(r2_score) for training = 90.73  
acc(relative error) for training = 9.95  
acc(rmse) for training = 14.67  

### training performance
acc(r2_score) for training = 98.7  
acc(relative error) for training = 3.69  
acc(rmse) for training = 5.5  
### Test performance
acc(r2_score) for training = 90.6  
acc(relative error) for training = 9.94  
acc(rmse) for training = 14.72  

### training performance
acc(r2_score) for training = 98.7  
acc(relative error) for training = 3.69  
acc(rmse) for training = 5.5  
### Test performance
acc(r2_score) for training = 90.59  
acc(relative error) for training = 9.98  
acc(rmse) for training = 14.74  

### training performance
acc(r2_score) for training = 98.69  
acc(relative error) for training = 3.69  
acc(rmse) for training = 5.5  
### Test performance
acc(r2_score) for training = 90.74  
acc(relative error) for training = 9.94  
acc(rmse) for training = 14.66  

### training performance
acc(r2_score) for training = 98.69  
acc(relative error) for training = 3.7  
acc(rmse) for training = 5.5  
### Test performance
acc(r2_score) for training = 90.77  
acc(relative error) for training = 9.92  
acc(rmse) for training = 14.65  

Avg R2 Score: 90.686  
Avg RMSE (normalized): 14.688 & in $ value: 2968.55  

### Test performance before pickling
acc(r2_score) for training = 90.78  
acc(relative error) for training = 9.84  
acc(rmse) for training = 14.64  
### Test performance after pickling
acc(r2_score) for training = 90.78  
acc(relative error) for training = 9.84  
acc(rmse) for training = 14.64  

## Training for Extra Trees Regressor starting ****

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.41  
### Test performance
acc(r2_score) for training = 90.13  
acc(relative error) for training = 10.25  
acc(rmse) for training = 15.14  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 89.96  
acc(relative error) for training = 10.24  
acc(rmse) for training = 15.21  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 89.96  
acc(relative error) for training = 10.28  
acc(rmse) for training = 15.23  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.42  
### Test performance
acc(r2_score) for training = 90.11  
acc(relative error) for training = 10.25  
acc(rmse) for training = 15.16  

### training performance
acc(r2_score) for training = 99.99  
acc(relative error) for training = 0.01  
acc(rmse) for training = 0.41  
### Test performance
acc(r2_score) for training = 90.14  
acc(relative error) for training = 10.22  
acc(rmse) for training = 15.15  

Avg R2 Score: 90.06  
Avg RMSE (normalized): 15.178 & in $ value: 3067.582  

### Test performance before pickling
acc(r2_score) for training = 90.18  
acc(relative error) for training = 10.14  
acc(rmse) for training = 15.11  
### Test performance after pickling
acc(r2_score) for training = 90.18  
acc(relative error) for training = 10.14  
acc(rmse) for training = 15.11  

## Training for Bagging Regressor starting ****

### training performance
acc(r2_score) for training = 98.22  
acc(relative error) for training = 4.07  
acc(rmse) for training = 6.42  
### Test performance
acc(r2_score) for training = 89.96  
acc(relative error) for training = 10.38  
acc(rmse) for training = 15.27  

### training performance
acc(r2_score) for training = 98.23  
acc(relative error) for training = 4.07  
acc(rmse) for training = 6.42  
### Test performance
acc(r2_score) for training = 89.84  
acc(relative error) for training = 10.35  
acc(rmse) for training = 15.3  

### training performance
acc(r2_score) for training = 98.23  
acc(relative error) for training = 4.06  
acc(rmse) for training = 6.4  
### Test performance
acc(r2_score) for training = 89.82  
acc(relative error) for training = 10.4  
acc(rmse) for training = 15.33  

### training performance
acc(r2_score) for training = 98.22  
acc(relative error) for training = 4.07  
acc(rmse) for training = 6.41  
### Test performance
acc(r2_score) for training = 90.03  
acc(relative error) for training = 10.34  
acc(rmse) for training = 15.22  

### training performance
acc(r2_score) for training = 98.22  
acc(relative error) for training = 4.07  
acc(rmse) for training = 6.42  
### Test performance
acc(r2_score) for training = 90.02  
acc(relative error) for training = 10.31  
acc(rmse) for training = 15.24  

Avg R2 Score: 89.934  
Avg RMSE (normalized): 15.272 & in $ value: 3086.58  

### Test performance before pickling
acc(r2_score) for training = 90.06  
acc(relative error) for training = 10.24  
acc(rmse) for training = 15.2  
### Test performance after pickling
acc(r2_score) for training = 90.06  
acc(relative error) for training = 10.24  
acc(rmse) for training = 15.2  

## Training for Gradient Boosting Regressor starting ****

### training performance
acc(r2_score) for training = 67.03  
acc(relative error) for training = 20.75  
acc(rmse) for training = 27.63  
### Test performance
acc(r2_score) for training = 66.91  
acc(relative error) for training = 20.82  
acc(rmse) for training = 27.72  

### training performance
acc(r2_score) for training = 67.01  
acc(relative error) for training = 20.82  
acc(rmse) for training = 27.67  
### Test performance
acc(r2_score) for training = 66.87  
acc(relative error) for training = 20.78  
acc(rmse) for training = 27.63  

### training performance
acc(r2_score) for training = 66.77  
acc(relative error) for training = 20.89  
acc(rmse) for training = 27.76  
### Test performance
acc(r2_score) for training = 66.49  
acc(relative error) for training = 20.92  
acc(rmse) for training = 27.82  

### training performance
acc(r2_score) for training = 66.38  
acc(relative error) for training = 20.98  
acc(rmse) for training = 27.9  
### Test performance
acc(r2_score) for training = 66.56  
acc(relative error) for training = 20.98  
acc(rmse) for training = 27.87  

### training performance
acc(r2_score) for training = 66.85  
acc(relative error) for training = 20.86  
acc(rmse) for training = 27.7  
### Test performance
acc(r2_score) for training = 67.04  
acc(relative error) for training = 20.84  
acc(rmse) for training = 27.69  

Avg R2 Score: 66.774  
Avg RMSE (normalized): 27.746 & in $ value: 5607.665  

### Test performance before pickling
acc(r2_score) for training = 66.97  
acc(relative error) for training = 20.71  
acc(rmse) for training = 27.71  
### Test performance after pickling
acc(r2_score) for training = 66.97  
acc(relative error) for training = 20.71  
acc(rmse) for training = 27.71  

## Training for Ada Boost Regressor starting ****


### training performance
acc(r2_score) for training = 73.05  
acc(relative error) for training = 21.15  
acc(rmse) for training = 24.98  
### Test performance
acc(r2_score) for training = 72.43  
acc(relative error) for training = 21.35  
acc(rmse) for training = 25.3  

### training performance
acc(r2_score) for training = 75.97  
acc(relative error) for training = 19.65  
acc(rmse) for training = 23.61  
### Test performance
acc(r2_score) for training = 75.34  
acc(relative error) for training = 19.73  
acc(rmse) for training = 23.84  

### training performance
acc(r2_score) for training = 73.55  
acc(relative error) for training = 20.92  
acc(rmse) for training = 24.77  
### Test performance
acc(r2_score) for training = 72.88  
acc(relative error) for training = 21.03  
acc(rmse) for training = 25.03  

### training performance
acc(r2_score) for training = 70.76  
acc(relative error) for training = 22.25  
acc(rmse) for training = 26.02  
### Test performance
acc(r2_score) for training = 70.22  
acc(relative error) for training = 22.39  
acc(rmse) for training = 26.3  

### training performance
acc(r2_score) for training = 76.8  
acc(relative error) for training = 19.19  
acc(rmse) for training = 23.18  
### Test performance
acc(r2_score) for training = 76.34  
acc(relative error) for training = 19.37  
acc(rmse) for training = 23.46  

Avg R2 Score: 73.442  
Avg RMSE (normalized): 24.786 & in $ value: 5009.428  

### Test performance before pickling
acc(r2_score) for training = 73.84  
acc(relative error) for training = 20.57  
acc(rmse) for training = 24.66  
### Test performance after pickling
acc(r2_score) for training = 73.84  
acc(relative error) for training = 20.57  
acc(rmse) for training = 24.66  


# Starting 5 Fold Cross Validation for Classifiers. Number of Class: 10

## Training for Nearest Neighbors Classifier starting ****

### training performance
Accuracy score for training = 57.08  
### Test performance
Accuracy score for training = 37.71  

### training performance
Accuracy score for training = 57.1  
### Test performance
Accuracy score for training = 37.67  

### training performance
Accuracy score for training = 57.05  
### Test performance
Accuracy score for training = 37.63  

### training performance
Accuracy score for training = 57.07  
### Test performance
Accuracy score for training = 37.7  

### training performance
Accuracy score for training = 57.12  
### Test performance
Accuracy score for training = 37.71  

Avg Accuracy Score: 37.684  

### Test performance before pickling
Accuracy score for training = 38.31  
### Test performance after pickling
Accuracy score for training = 38.31  

## Training for Decision Tree Classifier starting ****

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 47.57  

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 47.57  

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 47.85  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 47.39  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 47.51  

Avg Accuracy Score: 47.578  

### Test performance before pickling
Accuracy score for training = 48.18  
### Test performance after pickling
Accuracy score for training = 48.18  

## Training for Random Forest Classifier starting ****

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 53.13  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 53.38  

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 53.45  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 53.15  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 53.27  

Avg Accuracy Score: 53.276  

### Test performance before pickling
Accuracy score for training = 53.87  
### Test performance after pickling
Accuracy score for training = 53.87  

## Training for Extra Trees Classifier starting ****

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 52.8  

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 53.01  

### training performance
Accuracy score for training = 99.99  
### Test performance
Accuracy score for training = 53.1  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 53.05  

### training performance
Accuracy score for training = 99.98  
### Test performance
Accuracy score for training = 52.92  

Avg Accuracy Score: 52.976  

### Test performance before pickling
Accuracy score for training = 53.62  
### Test performance after pickling
Accuracy score for training = 53.62  

## Training for Neural Net Classifier starting ****

### training performance
Accuracy score for training = 23.93  
### Test performance
Accuracy score for training = 23.95  

### training performance
Accuracy score for training = 24.01  
### Test performance
Accuracy score for training = 24.07  

### training performance
Accuracy score for training = 23.98  
### Test performance
Accuracy score for training = 23.9  

### training performance
Accuracy score for training = 23.99  
### Test performance
Accuracy score for training = 24.02  

### training performance
Accuracy score for training = 23.95  
### Test performance
Accuracy score for training = 23.82  

Avg Accuracy Score: 23.952  

### Test performance before pickling
Accuracy score for training = 23.96  
### Test performance after pickling
Accuracy score for training = 23.96  

## Training for AdaBoost Classifier starting ****
