import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

# Import Historical Sample Data CSV
df = pd.read_csv("historical-sample-data.csv")
print(df)

# Create Pass Index
hist_data_samp = df
hist_data_samp = hist_data_samp[["Pass", "Outcome"]].set_index("Pass")
index_range = hist_data_samp.index.min(), hist_data_samp.index.max()
print(index_range)

# Graph Historical Sample Data
hist_data_samp.plot()
plt.xlabel("Passes Thrown")
plt.ylabel("Passes Caught")
_ = plt.title("Fantasy Football QB/WR - Historical Data")
# plt.show()

# Import Kernel Dependencies For Extrapolated Data
X = (hist_data_samp.index * 190).to_numpy().reshape(-1, 1)
y = hist_data_samp["Outcome"].to_numpy()
long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)
irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)
hist_data_samp_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
y_mean = y.mean()
gaussian_process = GaussianProcessRegressor(kernel=hist_data_samp_kernel, normalize_y=False)
gaussian_process.fit(X, y - y_mean)

# Fit Model With Extrapolated Data
X_test = np.linspace(start=1, stop=1990, num=1_990).reshape(-1, 1)
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
mean_y_pred += y_mean

# Graph Extrapolated Data
plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
plt.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2,
)
plt.legend()
plt.xlabel("Passes Thrown")
plt.ylabel("Passes Caught")
_ = plt.title(
    "Fantasy Football QB/WR - Forecasted Data"
)
plt.show()
