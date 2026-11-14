import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = pd.read_csv("C:/Users/ratho/Downloads/Student_Performance.csv")

print(dataset.describe().T.round(2).to_string())

print("First five rows:")
print(dataset.head())

print("\nDataset information:")
dataset.info()

print("\nMissing values:")
print(dataset.isnull().sum())

print("\nDuplicate rows:", dataset.duplicated().sum())

dataset = dataset.drop_duplicates()

dataset["Extracurricular Activities"] = dataset["Extracurricular Activities"].map(
    {"Yes": 1, "No": 0}
)

dataset = dataset.dropna()

print("\nDataset after cleaning:")
print(dataset.describe())

print("\nCorrelation with Performance Index:")
print(dataset.corr(numeric_only=True)["Performance Index"].sort_values(ascending=False))

plt.figure(figsize=(7, 4))
plt.hist(dataset["Performance Index"], bins=20, edgecolor="black")
plt.xlabel("Performance Index")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Performance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.scatter(dataset["Hours Studied"], dataset["Performance Index"])
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.title("Hours Studied vs Performance Index")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.scatter(dataset["Previous Scores"], dataset["Performance Index"])
plt.xlabel("Previous Scores")
plt.ylabel("Performance Index")
plt.title("Previous Scores vs Performance Index")
plt.tight_layout()
plt.show()

x = dataset[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
]

y = dataset["Performance Index"]

results = []

for test_size in [0.20, 0.25, 0.30]:

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    ratio = f"{int((1 - test_size) * 100)}:{int(test_size * 100)}"

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_prediction = linear_model.predict(x_test)

    polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = polynomial_features.fit_transform(x_train)
    x_test_poly = polynomial_features.transform(x_test)

    polynomial_model = LinearRegression()
    polynomial_model.fit(x_train_poly, y_train)
    polynomial_prediction = polynomial_model.predict(x_test_poly)

    tree_model = DecisionTreeRegressor(random_state=42, min_samples_leaf=5)
    tree_model.fit(x_train, y_train)
    tree_prediction = tree_model.predict(x_test)

    forest_model = RandomForestRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=2
    )
    forest_model.fit(x_train, y_train)
    forest_prediction = forest_model.predict(x_test)

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    knn_prediction = knn_model.predict(x_test)

    models = [
        ("Linear Regression", linear_prediction),
        ("Polynomial Regression", polynomial_prediction),
        ("Decision Tree", tree_prediction),
        ("Random Forest", forest_prediction),
        ("KNN", knn_prediction)
    ]

    print("\nTrain-test ratio:", ratio)

    for model_name, prediction in models:
        mse = mean_squared_error(y_test, prediction)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)

        print("\nModel:", model_name)
        print("MSE:", round(mse, 2))
        print("RMSE:", round(rmse, 2))
        print("MAE:", round(mae, 2))
        print("R2 Score:", round(r2, 4))

        results.append([
            ratio, model_name, mse, rmse, mae, r2
        ])

        if test_size == 0.20 and model_name == "Linear Regression":
            actual_values = y_test
            predicted_values = prediction

results_table = pd.DataFrame(
    results,
    columns=["Train-Test Ratio", "Model", "MSE", "RMSE", "MAE", "R2 Score"]
)

print("\nComplete model comparison:")
print(results_table.round(3).to_string(index=False))

results_table.to_csv("model_results.csv", index=False)
print("\nResults saved to model_results.csv")

plt.figure(figsize=(6, 4))
plt.scatter(actual_values, predicted_values)
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Performance (Linear Regression, 80:20)")
plt.tight_layout()
plt.show()

average_scores = results_table.groupby("Model")["R2 Score"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(average_scores.index, average_scores.values)
plt.xlabel("Model")
plt.ylabel("Average R2 Score")
plt.title("Average R2 Score of Five Models")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

print("\nBest model based on average R2 Score:")
print(average_scores.idxmax())
print("Average R2 Score:", round(average_scores.max(), 4))

print("\nProject completed successfully.")
