import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.tree import (DecisionTreeRegressor, plot_tree, export_text)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Problem 1
# Load Data
college_data=load_data("College")

# Separate Response from Features
y = college_data["Apps"]
X = college_data.drop(columns=["Apps"])

# Create 80/20 test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert categorical points to numerical points
X_train = pd.get_dummies(X_train, drop_first=True, dtype=int)
X_test  = pd.get_dummies(X_test, drop_first=True, dtype=int)

# Fit OLS model on training
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
# print(model.summary())

# Create predictions using testing set
predictions = model.predict(sm.add_constant(X_test))
# print(predictions)

# Calculate basic metrics
MSE       = mean_squared_error(y_test, predictions)
RMSE      = MSE ** 0.5
SS_Error  = ((y_test - predictions) ** 2).sum()
SS_T      = ((y_test - y_test.mean()) ** 2).sum()
R_Squared = 1 - (SS_Error / SS_T)
# print(MSE)
# print(RMSE)
# print(R_Squared)

# Fit Decision Tree on data
DTR_model = DecisionTreeRegressor(random_state=1)
DTR_model.fit(X_train, y_train)

# # Display tree (large)
# plt.figure(figsize=(12,12))
# plot_tree(DTR_model, feature_names=X_train.columns)
# plt.show()

# Create predictions for DTR_model
DTR_predictions = DTR_model.predict(X_test)
# print(DTR_predictions)

# Calculate basic metrics
DTR_MSE       = mean_squared_error(y_test, DTR_predictions)
DTR_RMSE      = DTR_MSE ** 0.5
DTR_SS_Error  = ((y_test - DTR_predictions) ** 2).sum()
DTR_R_Squared = 1 - (DTR_SS_Error / SS_T)
# print(DTR_MSE)
# print(DTR_RMSE)
# print(DTR_R_Squared)

# Calculate values of alpha that will prune individual branches, up until the last branch of the decision tree
path = DTR_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# --- This can cause overfitting ---
# Iterate over alpha values excluding the last alpha value since that will just be the entire tree pruned.
# all_models = []
# for alpha in ccp_alphas[:-1]:
#     temp_model = DecisionTreeRegressor(ccp_alpha=alpha, random_state=1)
#     temp_model.fit(X_train, y_train)
#     all_models.append(temp_model)
#
# Select best model by 
# best_model = min(all_models, key=lambda m: mean_squared_error(y_test, m.predict(X_test)))
# --- --- ---

# --- Better version as it does not apply alpha values directly to test sets and checks on training sets first ---
# Create a parameter grid for which arguments for Decision Tree Regressor models use as arguments (i.e. ccp_alpha is an argument for a DTR model)
param_grid = {"ccp_alpha": ccp_alphas}
grid       = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)

# Select best estimator
pruned_model = grid.best_estimator_

# Create predictions for pruned_DTR_model
pruned_DTR_predictions = pruned_model.predict(X_test)

# Calculate basic metrics
pruned_DTR_MSE       = mean_squared_error(y_test, pruned_DTR_predictions)
pruned_DTR_RMSE      = pruned_DTR_MSE ** 0.5
pruned_DTR_SS_Error  = ((y_test - pruned_DTR_predictions) ** 2).sum()
pruned_DTR_R_Squared = 1 - (pruned_DTR_SS_Error / SS_T)
# print(pruned_DTR_MSE)
# print(pruned_DTR_RMSE)
# print(pruned_DTR_R_Squared)

# Retrieving important features for pruned_DTR_model
pruned_DTR_important_features = pruned_model.feature_importances_

# # Visualizing feature importances
# pruned_DTR_feature_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': pruned_DTR_important_features
# }).sort_values(by='Importance', ascending=False)
# feature_df = pruned_DTR_feature_df[pruned_DTR_feature_df["Feature"] != 'const']
# feature_df.plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Name")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Importance")
# plt.show()

# Fit Bagging (random forest regressor) Models on data
bag_500_model  = RandomForestRegressor(max_features=len(X_train.columns), random_state=1, n_estimators=500)
bag_500_model.fit(X_train, y_train)
bag_1000_model = RandomForestRegressor(max_features=len(X_train.columns), random_state=1, n_estimators=1000)
bag_1000_model.fit(X_train, y_train)


# Create predictions for bag_500_model and bag_1000_model
bag_500_predictions  = bag_500_model.predict(X_test)
bag_1000_predictions = bag_1000_model.predict(X_test)

# Calculate basic metrics
bag_500_DTR_MSE       = mean_squared_error(y_test, bag_500_predictions)
bag_500_DTR_RMSE      = bag_500_DTR_MSE ** 0.5
bag_500_DTR_SS_Error  = ((y_test - bag_500_predictions) ** 2).sum()
bag_500_DTR_R_Squared = 1 - (bag_500_DTR_SS_Error / SS_T)
# print(bag_500_DTR_MSE)
# print(bag_500_DTR_RMSE)
# print(bag_500_DTR_R_Squared)
bag_1000_DTR_MSE       = mean_squared_error(y_test, bag_1000_predictions)
bag_1000_DTR_RMSE      = bag_1000_DTR_MSE ** 0.5
bag_1000_DTR_SS_Error  = ((y_test - bag_1000_predictions) ** 2).sum()
bag_1000_DTR_R_Squared = 1 - (bag_1000_DTR_SS_Error / SS_T)
# print(bag_1000_DTR_MSE)
# print(bag_1000_DTR_RMSE)
# print(bag_1000_DTR_R_Squared)

# Retrieving important features for pruned_DTR_model
bagged_500_important_features = bag_500_model.feature_importances_

# # Visualizing feature importances
# bagged_500_feature_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': bagged_500_important_features
# }).sort_values(by='Importance', ascending=False)
# feature_df = bagged_500_feature_df[bagged_500_feature_df["Feature"] != 'const']
# feature_df.plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Name")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Importance")
# plt.show()

# Fit Random Forest Regressors on data
random_forest_500_model  = RandomForestRegressor(max_features=3, random_state=1, n_estimators=500)
random_forest_500_model.fit(X_train, y_train)
random_forest_1000_model = RandomForestRegressor(max_features=3, random_state=1, n_estimators=1000)
random_forest_1000_model.fit(X_train, y_train)

# Create predictions for random_forest_500_model and random_forest_1000_model
random_forest_500_predictions  = random_forest_500_model.predict(X_test)
random_forest_1000_predictions = random_forest_1000_model.predict(X_test)

# Calculate basic metrics
random_forest_500_DTR_MSE       = mean_squared_error(y_test, random_forest_500_predictions)
random_forest_500_DTR_RMSE      = random_forest_500_DTR_MSE ** 0.5
random_forest_500_DTR_SS_Error  = ((y_test - random_forest_500_predictions) ** 2).sum()
random_forest_500_DTR_R_Squared = 1 - (random_forest_500_DTR_SS_Error / SS_T)
# print(random_forest_500_DTR_MSE)
# print(random_forest_500_DTR_RMSE)
# print(random_forest_500_DTR_R_Squared)
random_forest_1000_DTR_MSE       = mean_squared_error(y_test, random_forest_1000_predictions)
random_forest_1000_DTR_RMSE      = random_forest_1000_DTR_MSE ** 0.5
random_forest_1000_DTR_SS_Error  = ((y_test - random_forest_1000_predictions) ** 2).sum()
random_forest_1000_DTR_R_Squared = 1 - (random_forest_1000_DTR_SS_Error / SS_T)
# print(random_forest_1000_DTR_MSE)
# print(random_forest_1000_DTR_RMSE)
# print(random_forest_1000_DTR_R_Squared)

# Retrieving important features for pruned_DTR_model
random_forest_500_important_features = random_forest_500_model.feature_importances_

# # Visualizing feature importances
# random_forest_500_feature_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': random_forest_500_important_features
# }).sort_values(by='Importance', ascending=False)
# feature_df = random_forest_500_feature_df[random_forest_500_feature_df["Feature"] != 'const']
# feature_df.plot(kind='barh', x="Feature", y="Importance")
# plt.xlabel("Feature Name")
# plt.xticks(rotation=45)
# plt.ylabel("Feature Importance")
# plt.show()
