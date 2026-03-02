import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def main():
    # Create plots directory
    os.makedirs("plots", exist_ok=True)

    # 1. Data Loading
    print("Fetching Auto MPG dataset...")
    auto_mpg = fetch_ucirepo(id=9)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    df = pd.concat([X, y], axis=1)

    # 2.1 Exploratory Data Analysis
    print("\n--- 2.1 Exploratory Data Analysis ---")
    print("Missing values per feature:")
    print(df.isnull().sum())

    # Handle missing values (horsepower has missing values)
    # We'll drop rows with missing values for simplicity and robustness
    df_clean = df.dropna()
    X_clean = df_clean.drop("mpg", axis=1)
    y_clean = df_clean["mpg"]

    # Plot distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(y_clean, kde=True, bins=30)
    plt.title("Distribution of Target Variable (mpg)")
    plt.xlabel("mpg")
    plt.ylabel("Frequency")
    plt.savefig("plots/regression_target_distribution.png")
    plt.close()

    # Relationships between individual features and target
    features = X_clean.columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.scatterplot(x=X_clean[feature], y=y_clean, ax=axes[i], alpha=0.6)
        axes[i].set_title(f"mpg vs {feature}")
    # Remove empty subplot if any
    if len(features) < len(axes):
        for j in range(len(features), len(axes)):
            fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig("plots/regression_feature_target_relationships.png")
    plt.close()

    # Correlations among features
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_clean.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig("plots/regression_correlation_matrix.png")
    plt.close()

    # 2.2 Dataset and Preprocessing
    print("\n--- 2.2 Dataset and Preprocessing ---")
    # Split data into train, validation, and test sets (60% train, 20% val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )  # 0.25 * 0.8 = 0.2

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Helper function to evaluate models
    def evaluate_model(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} Performance:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        return mse, rmse, mae, r2

    # 2.3 Linear Regression Baseline
    print("\n--- 2.3 Linear Regression Baseline ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    y_val_pred_lr = lr_model.predict(X_val_scaled)
    evaluate_model(y_val, y_val_pred_lr, "Linear Regression (Validation)")

    # Prediction vs ground truth plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_val_pred_lr, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("Ground Truth (mpg)")
    plt.ylabel("Predictions (mpg)")
    plt.title("Linear Regression: Prediction vs Ground Truth")
    plt.savefig("plots/regression_lr_pred_vs_truth.png")
    plt.close()

    # 2.4 Polynomial Regression and Model Complexity
    print("\n--- 2.4 Polynomial Regression and Model Complexity ---")
    degrees = [1, 2, 3, 4, 5]
    train_errors = []
    val_errors = []

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_val_poly = poly_features.transform(X_val_scaled)

        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)

        y_train_pred = poly_model.predict(X_train_poly)
        y_val_pred = poly_model.predict(X_val_poly)

        train_errors.append(mean_squared_error(y_train, y_train_pred))
        val_errors.append(mean_squared_error(y_val, y_val_pred))

    # Model complexity curve
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, marker="o", label="Train MSE")
    plt.plot(degrees, val_errors, marker="s", label="Validation MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Polynomial Regression: Model Complexity Curve")
    plt.xticks(degrees)
    plt.yscale("log")  # Log scale for better visibility if errors explode
    plt.legend()
    plt.savefig("plots/regression_poly_complexity_curve.png")
    plt.close()

    best_degree = degrees[np.argmin(val_errors)]
    print(f"Best polynomial degree based on validation MSE: {best_degree}")

    # 2.5 K-Nearest Neighbors (KNN) Regression
    print("\n--- 2.5 K-Nearest Neighbors (KNN) Regression ---")
    k_values = list(range(1, 51))
    knn_train_errors = []
    knn_val_errors = []

    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)

        y_train_pred = knn.predict(X_train_scaled)
        y_val_pred = knn.predict(X_val_scaled)

        knn_train_errors.append(mean_squared_error(y_train, y_train_pred))
        knn_val_errors.append(mean_squared_error(y_val, y_val_pred))

    # KNN complexity curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, knn_train_errors, marker="o", label="Train MSE")
    plt.plot(k_values, knn_val_errors, marker="s", label="Validation MSE")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("KNN Regression: Train and Validation Error vs k")
    plt.legend()
    plt.savefig("plots/regression_knn_error_vs_k.png")
    plt.close()

    best_k = k_values[np.argmin(knn_val_errors)]
    print(f"Best k based on validation MSE: {best_k}")

    # Best KNN model prediction vs ground truth
    best_knn = KNeighborsRegressor(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_val_pred_knn = best_knn.predict(X_val_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, y_val_pred_knn, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
    plt.xlabel("Ground Truth (mpg)")
    plt.ylabel("Predictions (mpg)")
    plt.title(f"KNN (k={best_k}): Prediction vs Ground Truth")
    plt.savefig("plots/regression_knn_pred_vs_truth.png")
    plt.close()

    # 2.6 Optimization Behavior
    print("\n--- 2.6 Optimization Behavior (PyTorch) ---")

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        dtype=torch.float32,
    ).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(
        y_val.values if isinstance(y_val, pd.Series) else y_val, dtype=torch.float32
    ).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    def train_pytorch_model(batch_size, epochs=100, lr=0.01):
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model, Loss, Optimizer
        model = nn.Linear(X_train_tensor.shape[1], 1)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * X_batch.size(0)

            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
                val_losses.append(val_loss.item())

        return train_losses, val_losses

    # Train with Batch GD (batch_size = full dataset)
    bgd_train_loss, bgd_val_loss = train_pytorch_model(
        batch_size=len(train_dataset), epochs=100, lr=0.01
    )

    # Train with Mini-batch GD (batch_size = 32)
    mbgd_train_loss, mbgd_val_loss = train_pytorch_model(
        batch_size=32, epochs=100, lr=0.01
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(100), bgd_train_loss, label="Batch GD Train Loss")
    plt.plot(range(100), bgd_val_loss, label="Batch GD Val Loss")
    plt.plot(range(100), mbgd_train_loss, label="Mini-batch GD (32) Train Loss")
    plt.plot(range(100), mbgd_val_loss, label="Mini-batch GD (32) Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Optimization Behavior: Batch GD vs Mini-batch GD (PyTorch)")
    plt.legend()
    plt.savefig("plots/regression_optimization_behavior.png")
    plt.close()

    # 2.7 Model Comparison and Discussion
    print("\n--- 2.7 Model Comparison and Discussion ---")
    # Evaluate best models on Test Set
    print("Evaluating final models on TEST SET:")

    # Linear Regression
    y_test_pred_lr = lr_model.predict(X_test_scaled)
    print("\nLinear Regression Test Performance:")
    evaluate_model(y_test, y_test_pred_lr, "Linear Regression")

    # Polynomial Regression
    poly_features = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    best_poly_model = LinearRegression()
    best_poly_model.fit(X_train_poly, y_train)
    y_test_pred_poly = best_poly_model.predict(X_test_poly)
    print(f"\nPolynomial Regression (Degree {best_degree}) Test Performance:")
    evaluate_model(y_test, y_test_pred_poly, "Polynomial Regression")

    # KNN Regression
    y_test_pred_knn = best_knn.predict(X_test_scaled)
    print(f"\nKNN Regression (k={best_k}) Test Performance:")
    evaluate_model(y_test, y_test_pred_knn, "KNN Regression")

    print("\nScript execution completed. Plots are saved in the 'plots/' directory.")


if __name__ == "__main__":
    main()
