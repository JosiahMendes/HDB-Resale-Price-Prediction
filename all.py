from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as math
df = pd.read_csv("data/resale-flat-prices-sers-removed.csv", low_memory=False)

# %%
# Combine Multi-Generation and Multi Generation categories in flat type
df["flat_type"] = df["flat_type"].replace(
    "MULTI GENERATION", "MULTI-GENERATION")
df["flat_type"].unique()

# %%
# Clean flat model column by capitalising and renaming
df["flat_model"] = df["flat_model"].replace("2-room", "2-ROOM")
df["flat_model"] = df["flat_model"].replace("2-ROOM", "2 ROOM")
df["flat_model"] = df["flat_model"].replace("3Gen", "3 GEN")
df["flat_model"] = df["flat_model"].replace("Adjoined flat", "ADJOINED FLAT")
df["flat_model"] = df["flat_model"].replace("Apartment", "APARTMENT")
df["flat_model"] = df["flat_model"].replace("Improved", "IMPROVED")
df["flat_model"] = df["flat_model"].replace(
    "Improved-Maisonette", "IMPROVED-MAISONETTE")
df["flat_model"] = df["flat_model"].replace(
    "IMPROVED-MAISONETTE", "IMPROVED MAISONETTE")
df["flat_model"] = df["flat_model"].replace("Maisonette", "MAISONETTE")
df["flat_model"] = df["flat_model"].replace("Model A", "MODEL A")
df["flat_model"] = df["flat_model"].replace(
    "Model A-Maisonette", "MODEL A-MAISONETTE")
df["flat_model"] = df["flat_model"].replace(
    "MODEL A-MAISONETTE", "MODEL A MAISONETTE")
df["flat_model"] = df["flat_model"].replace("New Generation", "NEW GENERATION")
df["flat_model"] = df["flat_model"].replace("Model A2", "MODEL A2")
df["flat_model"] = df["flat_model"].replace(
    "MULTI GENERATION", "MULTI-GENERATION")
df["flat_model"] = df["flat_model"].replace(
    "Multi Generation", "MULTI-GENERATION")
df["flat_model"] = df["flat_model"].replace(
    "Premium Apartment", "PREMIUM APARTMENT")
df["flat_model"] = df["flat_model"].replace(
    "Premium Apartment Loft", "PREMIUM APARTMENT LOFT")
df["flat_model"] = df["flat_model"].replace(
    "Premium Maisonette", "PREMIUM MAISONETTE")
df["flat_model"] = df["flat_model"].replace("Simplified", "SIMPLIFIED")
df["flat_model"] = df["flat_model"].replace("Standard", "STANDARD")
df["flat_model"] = df["flat_model"].replace("Terrace", "TERRACE")
df["flat_model"] = df["flat_model"].replace("Type S1", "TYPE S1")
df["flat_model"] = df["flat_model"].replace("Type S2", "TYPE S2")

df["flat_model"].unique()

# Convert lease_commence_date to remaining lease at point of transaction
# lease commencing 1976 and sale in 1990 =  1976+99-1990 = 85 years remaining
df["remaining_lease"] = (df["lease_commence_date"] +
                         99 - df["month"].str[:4].astype(int))/99
df = df.drop(columns=['lease_commence_date'])

# Calculate inflated_adjusted_price
hdb_resale_price_index = pd.read_csv(
    "data/housing-and-development-board-resale-price-index-1q2009-100-monthly.csv", low_memory=False, index_col=0)
current_index = hdb_resale_price_index.tail(1)["index"].values[0]

# join with hdb_resale_price_index
df = df.join(hdb_resale_price_index, on="month", how="left", rsuffix="_index")
df["adjusted_price"] = df["resale_price"] * (current_index / df["index"])
df.drop(columns=["index", "resale_price", "month"], inplace=True)


# Add Region Data
# Central = ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TOA PAYOH']
# East = ['BEDOK','PASIR RIS', 'TAMPINES']
# West = ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 'JURONG EAST', 'JURONG WEST']
# North East = ['ANG MO KIO','HOUGANG', 'PUNGGOL', 'SENGKANG','SERANGOON']
# North = ['SEMBAWANG', 'WOODLANDS', 'YISHUN',]

def get_region(row):
    town = row["town"]
    if town in ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TOA PAYOH']:
        return "CENTRAL"
    elif town in ['BEDOK', 'PASIR RIS', 'TAMPINES']:
        return "EAST"
    elif town in ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 'JURONG EAST', 'JURONG WEST']:
        return "WEST"
    elif town in ['ANG MO KIO', 'HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON']:
        return "NORTH-EAST"
    elif town in ['SEMBAWANG', 'WOODLANDS', 'YISHUN']:
        return "NORTH"


df["region"] = df.apply(get_region, axis=1)


# Add coordinate data from block_street_name_coords.json
df["block_street_name"] = df["block"].astype(
    str) + " " + df["street_name"].astype(str)
df = df.drop(columns=['block', 'street_name'])
block_street_name_coords = pd.read_json("data/block_street_name_coords.json")
block_street_name_coords = block_street_name_coords.transpose()

df = df.join(block_street_name_coords, on="block_street_name",
             how="left", rsuffix="_coords")
df.drop(columns=["block_street_name"], inplace=True)


# Add distance from Downtown Core planning area (CBD)
# Downtown Core planning area (CBD) = 1.286667, 103.853611
dg_mrt_lat = np.radians(1.286667)
dg_mrt_long = np.radians(103.853611)

df['distance_from_cbd'] = 6367 * 2 * np.arcsin(np.sqrt(np.sin((np.radians(df['latitude']) - dg_mrt_lat)/2)**2 + math.cos(
    math.radians(37.2175900)) * np.cos(np.radians(df['latitude'])) * np.sin((np.radians(df['longitude']) - dg_mrt_long)/2)**2))


# Transform storey_range to median of range (e.g. 01 TO 03 = 2)
def convert_to_median(row):
    storey_range = row["storey_range"].split(" TO ")
    median = (int(storey_range[0]) + int(storey_range[1])) / 2
    return median


df["median_storey"] = df.apply(convert_to_median, axis=1)
df = df.drop(columns=['storey_range'])


# Transform flat_type to ordinal encoding, town and flat_model to one-hot encoding
cols = ["town", "region", "longitude", "latitude", "distance_from_cbd",
        "flat_type", "flat_model", "floor_area_sqm", "median_storey", "remaining_lease",
        "adjusted_price"
        ]
df = df[cols]

pipeline = ColumnTransformer([
    # Add normalisation for numerical columns
    # ("s", StandardScaler(), ["longitude", "latitude", "distance_from_cbd", "floor_area_sqm", "median_storey", "remaining_lease"]),
    ("o", OrdinalEncoder(), ["flat_type"]),
    ("n", OneHotEncoder(sparse_output=False),
     ["town", "flat_model", "region"]),
], remainder='passthrough', verbose_feature_names_out=False)
pipeline.set_output(transform="pandas")
df = pipeline.fit_transform(df)

# %% [markdown]
# # Split Train-Test-Validation

# %%
# Split dataset randomly into 80% training and 20% test, then split training into 80% training and 20% validation
train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)

# %%
# Validate that the split is correct
print("Train size: ", len(train))
print("Validation size: ", len(val))
print("Test size: ", len(test))

# Validate that the split is a representative sample of the original dataset
print("Train mean: ", train["adjusted_price"].mean())
print("Validation mean: ", val["adjusted_price"].mean())
print("Test mean: ", test["adjusted_price"].mean())


# %%
# Split into X and y
X_train = train.drop(columns=["adjusted_price"])
y_train = train["adjusted_price"]

X_val = val.drop(columns=["adjusted_price"])
y_val = val["adjusted_price"]

X_test = test.drop(columns=["adjusted_price"])
y_test = test["adjusted_price"]

# %% [markdown]
# # Training Models

# %%
# Get MSE, MAE and MPE for model where we predict the mean of the training set

# dataset_mean = df["adjusted_price"].mean()
# dataset_median = df["adjusted_price"].median()

# mean = y_train.median()
# y_pred = np.full(len(y_val), mean)

# print("MSE: ", mean_squared_error(y_val, y_pred))
# print("MAE: ", mean_absolute_error(y_val, y_pred))
# print("MPE: ", mean_absolute_percentage_error(y_val, y_pred))
# print("Mean: ", dataset_mean)
# print("Median: ", dataset_median)

# %%
# # Calculate f_regression scores for each feature
# from sklearn.feature_selection import f_regression

# # Sort feature names by f_regression scores
# feature_names = X_train.columns
# scores = f_regression(X_train, y_train)[0]
# feature_scores = pd.DataFrame({"feature": feature_names, "score": scores})
# feature_scores.sort_values(by="score", ascending=False)


# %%
# Train Linear Regression and evaluate on validation set
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # Print MSE and MAE
# y_val_pred = lin_reg.predict(X_val)
# print("MSE for linear regression model 1 =>", mean_squared_error(y_val, y_val_pred))
# print("MAE for linear regression model 1 =>", mean_absolute_error(y_val, y_val_pred))
# print("MAPE for linear regression model 1 =>", mean_absolute_percentage_error(y_val, y_val_pred))


# %%
# Train Ridge Regression and evalue on validation set

# rr = Ridge(alpha=1).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 1 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 1 =>",
#       mean_absolute_error(y_val, y_val_pred))

# rr = Ridge(alpha=10).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 10 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 10 =>",
#       mean_absolute_error(y_val, y_val_pred))

# rr = Ridge(alpha=100).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 100 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 100 =>",
#       mean_absolute_error(y_val, y_val_pred))

# rr = Ridge(alpha=1000).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 1000 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 1000 =>",
#       mean_absolute_error(y_val, y_val_pred))

# rr = Ridge(alpha=10000).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 10000 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 10000 =>",
#       mean_absolute_error(y_val, y_val_pred))

# rr = Ridge(alpha=100000).fit(X_train, y_train)

# y_val_pred = rr.predict(X_val)
# print("MSE for ridge regression alpha 10000 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for ridge regression alpha 10000 =>",
#       mean_absolute_error(y_val, y_val_pred))


# %%
# # Train support vector regression and evaluate on validation set
# from sklearn.svm import SVR

# svr = SVR(kernel="linear", C=1, epsilon=0.1)
# svr.fit(X_train, y_train)

# y_val_pred = svr.predict(X_val)
# print("MSE for SVR =>", mean_squared_error(y_val, y_val_pred))
# print("MAE for SVR =>", mean_absolute_error(y_val, y_val_pred))

# %%
# Train Decison Tree Regressor and evaluate on validation set
# from sklearn.tree import DecisionTreeRegressor

# tree_reg = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# y_val_pred = tree_reg.predict(X_val)
# print("MSE for decision tree regression model 1 =>", mean_squared_error(y_val, y_val_pred))
# print("MAE for decision tree regression model 1 =>", mean_absolute_error(y_val, y_val_pred))
# print("MAPE for decision tree regression model 1 =>", mean_absolute_percentage_error(y_val, y_val_pred))

# print(tree_reg.get_depth())

# %%
# train using k-nearest neighbors
# from sklearn.neighbors import KNeighborsRegressor

# knn_reg = KNeighborsRegressor(weights="distance", n_jobs=8).fit(X_train, y_train)

# y_val_pred = knn_reg.predict(X_val)
# print("MSE for k-nearst neighbors regression model 1 =>", mean_squared_error(y_val, y_val_pred))
# print("MAE for k-nearst neighbors regression model 1 =>", mean_absolute_error(y_val, y_val_pred))
# print("MAPE for k-nearst neighbors regression model 1 =>", mean_absolute_percentage_error(y_val, y_val_pred))

# %%
# # Train Linear support vector regressor and evaluate on validation set
# from sklearn.svm import LinearSVR

# svm_reg = LinearSVR(random_state=42).fit(X_train, y_train)

# y_val_pred = svm_reg.predict(X_val)
# print("MSE for linear support vector regression model 1 =>", mean_squared_error(y_val, y_val_pred))
# print("MAE for linear support vector regression model 1 =>", mean_absolute_error(y_val, y_val_pred))

# %%
# Train Random Forest Regressor and evaluate on validation set

# forest_reg = RandomForestRegressor(
#     n_estimators=100, random_state=42, n_jobs=8).fit(X_train, y_train)

# y_val_pred = forest_reg.predict(X_val)
# print("MSE for random forest regression model 1 =>",
#       mean_squared_error(y_val, y_val_pred))
# print("MAE for random forest regression model 1 =>",
#       mean_absolute_error(y_val, y_val_pred))
# print("MAPE for random forest regression model 1 =>",
#   mean_absolute_percentage_error(y_val, y_val_pred))

# %% [markdown]
# # Tune Random Forest Regressor

# %%
# Use grid search to find best hyperparameters for random forest regressor

param_grid = [
    {'n_estimators': [3, 10, 30, 100]},
]
regr = RandomForestRegressor(random_state=0)
grid_search = GridSearchCV(regr, param_grid, cv=2, verbose=2,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_estimator_)


# %%
# Visualise adjusted_price per sqm on Singapore map
# import matplotlib.colors as colors
def visualise(df, vmin, vmax):

    df_sorted = df.sort_values(by='price_per_sqm')
    x = df_sorted['longitude']
    y = df_sorted['latitude']
    c = df_sorted['price_per_sqm']

    plt.rcParams['figure.figsize'] = [20, 10]
    plt.rcParams['figure.dpi'] = 100

    # add image of singapore map
    img = plt.imread(
        'data/3247px-Singapore_location_map_(main_island).svg.png')
    plt.imshow(img, extent=[103.557, 104.131, 1.129, 1.493])

    # set axes limits
    plt.xlim(103.62, 104.03)
    plt.ylim(1.23, 1.465)

    # Set axes titles
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.scatter(x, y, s=0.01, c=c, cmap='OrRd',
                norm=colors.Normalize(vmin=vmin, vmax=vmax), alpha=0.8)
    cbar = plt.colorbar()
    cbar.set_label('Price per sqm', rotation=270, labelpad=20)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# df2 = df
# df2["price_per_sqm"] = df2["adjusted_price"] / df2["floor_area_sqm"]
# visualise(df2, df2["price_per_sqm"].quantile(0.10), df2["price_per_sqm"].quantile(0.90))

# %% [markdown]
# ###
