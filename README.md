# Retail_customer_dataset-to-predict-Churn

# Hyperparameter tuning grid for Logistic Regression in scikit-learn, typically used with GridSearchCV or RandomizedSearchCV.
Explanation of Each Parameter
clf__penalty: ['l2']
Penalty refers to the type of regularization applied to prevent overfitting. 'l2' = Ridge regularization (most common for Logistic Regression). Other options (not in this grid): 'l1' (Lasso), 'elasticnet'

clf__C: [0.01, 0.1, 1, 10, 100]
C is the inverse of regularization strength. Smaller C → stronger regularization (simpler model). Larger C → weaker regularization (model fits data more closely). This grid tests a wide range from very strong regularization (0.01) to almost no regularization (100)

clf__solver: ['lbfgs', 'liblinear']
Solver is the algorithm used to optimize the logistic regression cost function. 'lbfgs': Good for large datasets, supports L2 penalty. 'liblinear': Works well for small datasets, supports L1 and L2 penalties

Why tune these?
Different penalties and solvers affect convergence and performance.
C controls bias-variance tradeoff.
Choosing the right combination improves accuracy and generalization.
