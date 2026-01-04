#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , Normalizer 
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
import sklearn.decomposition as dec
from sklearn.linear_model import SGDRegressor , Ridge , LinearRegression , Lasso , LassoLars ,RANSACRegressor, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor , RandomForestRegressor , GradientBoostingRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics
# import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager
import random
from lazypredict.Supervised import LazyRegressor
from sklearn.svm import SVR


# In[3]:


df = pd.read_excel(r"D:\Articles\Elmi\Final.xlsx", sheet_name='Dataset', header=0, engine='openpyxl')
y = df.iloc[:, 6].to_numpy().reshape((-1, 1))
X = df.iloc[:, [0,1,2,3,4,5,]].to_numpy()

# === مدل اولیه: بدون Grid Search ===
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=7)

# === تنظیمات گرافیکی عمومی ===
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)


# In[7]:


def plot_model_results(ytr, yprtr, yte, yprte, title="Model", name="fig"):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    
    ytr = np.ravel(ytr)
    yprtr = np.ravel(yprtr)
    yte = np.ravel(yte)
    yprte = np.ravel(yprte)
    
    a, b = 0, max(np.max(ytr), np.max(yte), np.max(yprtr), np.max(yprte)) + 10

    # --- metrics ---
    r2_tr = round(r2_score(ytr, yprtr), 2)
    rmse_tr = round(mean_squared_error(ytr, yprtr, squared=False), 2)
    mae_tr = round(mean_absolute_error(ytr, yprtr), 2)

    r2_te = round(r2_score(yte, yprte), 2)
    rmse_te = round(mean_squared_error(yte, yprte, squared=False), 2)
    mae_te = round(mean_absolute_error(yte, yprte), 2)

    # =========================
    # Prediction plot
    # =========================
    axs[0].scatter(ytr, yprtr, s=60, edgecolors='black',
                   facecolors='#9ecae1', marker='o', label='Train')
    axs[0].scatter(yte, yprte, s=60, edgecolors='black',
                   facecolors='#fc9272', marker='^', label='Test')

    x_line = np.linspace(a, b, 200)

    # y = x
    axs[0].plot(x_line, x_line, '--', color='gray', label='y = x')

    # ±10% error bands
    axs[0].plot(x_line, 1.1 * x_line, ':', color='black', linewidth=1,
                label='+10%')
    axs[0].plot(x_line, 0.9 * x_line, ':', color='black', linewidth=1,
                label='-10%')

    axs[0].set_xlabel("Bond strength (MPa)_Experimental")
    axs[0].set_ylabel("Bond strength (MPa)_Predicted")
    axs[0].set_title(f"{title} - Prediction")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    axs[0].text(
        0.95, 0.05,
        f"Train R² = {r2_tr}, RMSE = {rmse_tr}, MAE = {mae_tr}\n"
        f"Test  R² = {r2_te}, RMSE = {rmse_te}, MAE = {mae_te}",
        transform=axs[0].transAxes,
        fontsize=11,
        va='bottom',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
    )

    # =========================
    # Residual plot (unchanged)
    # =========================
    res_tr = ytr - yprtr
    res_te = yte - yprte

    axs[1].scatter(ytr, res_tr, s=60, edgecolors='black',
                   facecolors='#9ecae1', marker='o', label='Train')
    axs[1].scatter(yte, res_te, s=60, edgecolors='black',
                   facecolors='#fc9272', marker='^', label='Test')
    axs[1].axhline(0, linestyle='--', color='gray')

    axs[1].set_xlabel("True Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_ylim(-b * 0.6, b * 0.6)
    axs[1].set_title(f"{title} - Residual")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"D:\\Articles\\Elmi\\Figs\\{name}.png",
        dpi=1000,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()


# In[5]:


model = DecisionTreeRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)

# --- ارزیابی و نمایش ---
print("===== مدل اولیه =====")
print(f"Train R²: {round(r2_score(ytr, yprtr), 2)}, RMSE: {round(mean_squared_error(ytr, yprtr, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr, yprtr), 2)}")
print(f"Test  R²: {round(r2_score(yte, yprte), 2)}, RMSE: {round(mean_squared_error(yte, yprte, squared=False), 2)}, MAE: {round(mean_absolute_error(yte, yprte), 2)}")
plot_model_results(ytr, yprtr, yte, yprte, title="Before Grid Search", name="before_grid_search")

# === مدل دوم: با Grid Search ===
Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, train_size=0.7, random_state=7)
model1 = DecisionTreeRegressor(
    random_state=0,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=5,
    max_features='sqrt'
)
model1.fit(Xtr1, ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)

# --- ارزیابی و نمایش ---
print("===== مدل بهینه‌سازی شده =====")
print(f"Train R²: {round(r2_score(ytr1, yprtr1), 2)}, RMSE: {round(mean_squared_error(ytr1, yprtr1, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr1, yprtr1), 2)}")
print(f"Test  R²: {round(r2_score(yte1, yprte1), 2)}, RMSE: {round(mean_squared_error(yte1, yprte1, squared=False), 2)}, MAE: {round(mean_absolute_error(yte1, yprte1), 2)}")
plot_model_results(ytr1, yprtr1, yte1, yprte1, title="After Grid Search", name="after_grid_search")


# In[ ]:


model = DecisionTreeRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)

# --- ارزیابی و نمایش ---
print("===== مدل اولیه =====")
print(f"Train R²: {round(r2_score(ytr, yprtr), 2)}, RMSE: {round(mean_squared_error(ytr, yprtr, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr, yprtr), 2)}")
print(f"Test  R²: {round(r2_score(yte, yprte), 2)}, RMSE: {round(mean_squared_error(yte, yprte, squared=False), 2)}, MAE: {round(mean_absolute_error(yte, yprte), 2)}")
plot_model_results(ytr, yprtr, yte, yprte, title="Before Grid Search", name="before_grid_search")

# === مدل دوم: با Grid Search ===
Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, train_size=0.7, random_state=7)
model1 = DecisionTreeRegressor(
    random_state=0,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=5,
    max_features='sqrt'
)
model1.fit(Xtr1, ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)

# --- ارزیابی و نمایش ---
print("===== مدل بهینه‌سازی شده =====")
print(f"Train R²: {round(r2_score(ytr1, yprtr1), 2)}, RMSE: {round(mean_squared_error(ytr1, yprtr1, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr1, yprtr1), 2)}")
print(f"Test  R²: {round(r2_score(yte1, yprte1), 2)}, RMSE: {round(mean_squared_error(yte1, yprte1, squared=False), 2)}, MAE: {round(mean_absolute_error(yte1, yprte1), 2)}")
plot_model_results(ytr1, yprtr1, yte1, yprte1, title="After Grid Search", name="after_grid_search")


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def plot_model_results(ytr, yprtr, yte, yprte, title="Model", name="fig"):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    
    ytr = np.ravel(ytr)
    yprtr = np.ravel(yprtr)
    yte = np.ravel(yte)
    yprte = np.ravel(yprte)
    
    a, b = 0, max(np.max(ytr), np.max(yte), np.max(yprtr), np.max(yprte)) + 10

    # --- metrics ---
    r2_tr = round(r2_score(ytr, yprtr), 2)
    rmse_tr = round(mean_squared_error(ytr, yprtr, squared=False), 2)
    mae_tr = round(mean_absolute_error(ytr, yprtr), 2)

    r2_te = round(r2_score(yte, yprte), 2)
    rmse_te = round(mean_squared_error(yte, yprte, squared=False), 2)
    mae_te = round(mean_absolute_error(yte, yprte), 2)

    # =========================
    # Prediction plot
    # =========================
    axs[0].scatter(ytr, yprtr, s=60, edgecolors='black',
                   facecolors='orange', marker='o', label='Train')
    axs[0].scatter(yte, yprte, s=60, edgecolors='black',
                   facecolors='purple', marker='^', label='Test')

    x_line = np.linspace(a, b, 200)

    # y = x
    axs[0].plot(x_line, x_line, '--', color='gray', label='y = x')

    # ±10% error bands
    axs[0].plot(x_line, 1.1 * x_line, ':', color='black', linewidth=1,
                label='+10%')
    axs[0].plot(x_line, 0.9 * x_line, ':', color='black', linewidth=1,
                label='-10%')

    axs[0].set_xlabel("Bond strength (MPa)_Experimental")
    axs[0].set_ylabel("Bond strength (MPa)_Predicted")
    axs[0].set_title(f"{title} - Prediction")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    axs[0].text(
        0.95, 0.05,
        f"Train R² = {r2_tr}, RMSE = {rmse_tr}, MAE = {mae_tr}\n"
        f"Test  R² = {r2_te}, RMSE = {rmse_te}, MAE = {mae_te}",
        transform=axs[0].transAxes,
        fontsize=11,
        va='bottom',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
    )

    # =========================
    # Residual plot (unchanged)
    # =========================
    res_tr = ytr - yprtr
    res_te = yte - yprte

    axs[1].scatter(ytr, res_tr, s=60, edgecolors='black',
                   facecolors='orange', marker='o', label='Train')
    axs[1].scatter(yte, res_te, s=60, edgecolors='black',
                   facecolors='purple', marker='^', label='Test')
    axs[1].axhline(0, linestyle='--', color='gray')

    axs[1].set_xlabel("True Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_ylim(-b * 0.6, b * 0.6)
    axs[1].set_title(f"{title} - Residual")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"D:\\Articles\\Elmi\\Figs\\{name}.png",
        dpi=1000,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

# === تنظیمات گرافیکی عمومی ===
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)

# === مدل اولیه: بدون Grid Search ===
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=7)
model = RandomForestRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)

# --- ارزیابی و نمایش ---
print("===== مدل اولیه =====")
print(f"Train R²: {round(r2_score(ytr, yprtr), 2)}, RMSE: {round(mean_squared_error(ytr, yprtr, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr, yprtr), 2)}")
print(f"Test  R²: {round(r2_score(yte, yprte), 2)}, RMSE: {round(mean_squared_error(yte, yprte, squared=False), 2)}, MAE: {round(mean_absolute_error(yte, yprte), 2)}")
plot_model_results(ytr, yprtr, yte, yprte, title="Before Grid Search")

# === مدل دوم: با Grid Search ===
Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, train_size=0.7, random_state=7)
model1 = RandomForestRegressor(random_state=0, 
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              max_depth=None, 
                              max_features='auto', 
                              n_estimators=50)
model1.fit(Xtr1, ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)

# --- ارزیابی و نمایش ---
print("===== مدل بهینه‌سازی شده =====")
print(f"Train R²: {round(r2_score(ytr1, yprtr1), 2)}, RMSE: {round(mean_squared_error(ytr1, yprtr1, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr1, yprtr1), 2)}")
print(f"Test  R²: {round(r2_score(yte1, yprte1), 2)}, RMSE: {round(mean_squared_error(yte1, yprte1, squared=False), 2)}, MAE: {round(mean_absolute_error(yte1, yprte1), 2)}")
plot_model_results(ytr1, yprtr1, yte1, yprte1, title="After Grid Search")


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

def plot_model_results(ytr, yprtr, yte, yprte, title="Model", name="fig"):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    
    ytr = np.ravel(ytr)
    yprtr = np.ravel(yprtr)
    yte = np.ravel(yte)
    yprte = np.ravel(yprte)
    
    a, b = 0, max(np.max(ytr), np.max(yte), np.max(yprtr), np.max(yprte)) + 10

    # --- metrics ---
    r2_tr = round(r2_score(ytr, yprtr), 2)
    rmse_tr = round(mean_squared_error(ytr, yprtr, squared=False), 2)
    mae_tr = round(mean_absolute_error(ytr, yprtr), 2)

    r2_te = round(r2_score(yte, yprte), 2)
    rmse_te = round(mean_squared_error(yte, yprte, squared=False), 2)
    mae_te = round(mean_absolute_error(yte, yprte), 2)

    # =========================
    # Prediction plot
    # =========================
    axs[0].scatter(ytr, yprtr, s=60, edgecolors='black',
                   facecolors='pink', marker='o', label='Train')
    axs[0].scatter(yte, yprte, s=60, edgecolors='black',
                   facecolors='cyan', marker='^', label='Test')

    x_line = np.linspace(a, b, 200)

    # y = x
    axs[0].plot(x_line, x_line, '--', color='gray', label='y = x')

    # ±10% error bands
    axs[0].plot(x_line, 1.1 * x_line, ':', color='black', linewidth=1,
                label='+10%')
    axs[0].plot(x_line, 0.9 * x_line, ':', color='black', linewidth=1,
                label='-10%')

    axs[0].set_xlabel("Bond strength (MPa)_Experimental")
    axs[0].set_ylabel("Bond strength (MPa)_Predicted")
    axs[0].set_title(f"{title} - Prediction")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    axs[0].text(
        0.95, 0.05,
        f"Train R² = {r2_tr}, RMSE = {rmse_tr}, MAE = {mae_tr}\n"
        f"Test  R² = {r2_te}, RMSE = {rmse_te}, MAE = {mae_te}",
        transform=axs[0].transAxes,
        fontsize=11,
        va='bottom',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
    )

    # =========================
    # Residual plot (unchanged)
    # =========================
    res_tr = ytr - yprtr
    res_te = yte - yprte

    axs[1].scatter(ytr, res_tr, s=60, edgecolors='black',
                   facecolors='pink', marker='o', label='Train')
    axs[1].scatter(yte, res_te, s=60, edgecolors='black',
                   facecolors='cyan', marker='^', label='Test')
    axs[1].axhline(0, linestyle='--', color='gray')

    axs[1].set_xlabel("True Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_ylim(-b * 0.6, b * 0.6)
    axs[1].set_title(f"{title} - Residual")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"D:\\Articles\\Elmi\\Figs\\{name}.png",
        dpi=1000,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

# === تنظیمات گرافیکی عمومی ===
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)
# === تنظیمات گرافیکی عمومی ===
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)

# === مدل اولیه: بدون Grid Search ===
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=7)
model = CatBoostRegressor(random_state=0)
model.fit(Xtr, ytr)
yprtr = model.predict(Xtr)
yprte = model.predict(Xte)

# --- ارزیابی و نمایش ---
print("===== مدل اولیه =====")
print(f"Train R²: {round(r2_score(ytr, yprtr), 2)}, RMSE: {round(mean_squared_error(ytr, yprtr, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr, yprtr), 2)}")
print(f"Test  R²: {round(r2_score(yte, yprte), 2)}, RMSE: {round(mean_squared_error(yte, yprte, squared=False), 2)}, MAE: {round(mean_absolute_error(yte, yprte), 2)}")
plot_model_results(ytr, yprtr, yte, yprte, title="Before Grid Search")

# === مدل دوم: با Grid Search ===
Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, train_size=0.7, random_state=7)
model1 = CatBoostRegressor(
                depth=4,
                learning_rate=0.05,
                iterations=500,
                random_state=0,
                verbose=0
            )
model1.fit(Xtr1, ytr1)
yprtr1 = model1.predict(Xtr1)
yprte1 = model1.predict(Xte1)

# --- ارزیابی و نمایش ---
print("===== مدل بهینه‌سازی شده =====")
print(f"Train R²: {round(r2_score(ytr1, yprtr1), 2)}, RMSE: {round(mean_squared_error(ytr1, yprtr1, squared=False), 2)}, MAE: {round(mean_absolute_error(ytr1, yprtr1), 2)}")
print(f"Test  R²: {round(r2_score(yte1, yprte1), 2)}, RMSE: {round(mean_squared_error(yte1, yprte1, squared=False), 2)}, MAE: {round(mean_absolute_error(yte1, yprte1), 2)}")
plot_model_results(ytr1, yprtr1, yte1, yprte1, title="After Grid Search")


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =========================
# Load data
# =========================
# فرض بر این است که X و y قبلاً ساخته شده‌اند
# X = df.iloc[:, [0,1,2,3,4,5]].values
# y = df.iloc[:, 6].values

# =========================
# Plot settings
# =========================
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)

# =========================
# Train / Test split
# =========================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=0.7, random_state=7
)

# =========================
# Scaling (StandardScaler)
# =========================
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

# =====================================================
# Function: Prediction + Residual + ±10% bands
# =====================================================
def plot_model_results(ytr, yprtr, yte, yprte, title, name,
                       train_color, test_color):

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ytr = np.ravel(ytr)
    yprtr = np.ravel(yprtr)
    yte = np.ravel(yte)
    yprte = np.ravel(yprte)

    a, b = 0, max(np.max(ytr), np.max(yte),
                  np.max(yprtr), np.max(yprte)) + 10

    r2_tr = round(r2_score(ytr, yprtr), 2)
    rmse_tr = round(mean_squared_error(ytr, yprtr, squared=False), 2)
    mae_tr = round(mean_absolute_error(ytr, yprtr), 2)

    r2_te = round(r2_score(yte, yprte), 2)
    rmse_te = round(mean_squared_error(yte, yprte, squared=False), 2)
    mae_te = round(mean_absolute_error(yte, yprte), 2)

    # ================= Prediction plot =================
    axs[0].scatter(ytr, yprtr, s=60, edgecolors='black',
                   facecolors=train_color, label='Train')
    axs[0].scatter(yte, yprte, s=60, edgecolors='black',
                   facecolors=test_color, marker='^', label='Test')

    x = np.linspace(a, b, 200)
    axs[0].plot(x, x, '--', color='gray', label='y = x')
    axs[0].plot(x, 1.1 * x, ':', color='black', linewidth=1, label='+10%')
    axs[0].plot(x, 0.9 * x, ':', color='black', linewidth=1, label='-10%')

    axs[0].set_xlabel("Bond strength (MPa)_Experimental")
    axs[0].set_ylabel("Bond strength (MPa)_Predicted")
    axs[0].set_title(f"{title} - Prediction")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    axs[0].text(
        0.95, 0.05,
        f"Train R² = {r2_tr}, RMSE = {rmse_tr}, MAE = {mae_tr}\n"
        f"Test  R² = {r2_te}, RMSE = {rmse_te}, MAE = {mae_te}",
        transform=axs[0].transAxes,
        fontsize=11,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', edgecolor='gray')
    )

    # ================= Residual plot =================
    res_tr = ytr - yprtr
    res_te = yte - yprte

    axs[1].scatter(ytr, res_tr, s=60, edgecolors='black',
                   facecolors=train_color, label='Train')
    axs[1].scatter(yte, res_te, s=60, edgecolors='black',
                   facecolors=test_color, marker='^', label='Test')
    axs[1].axhline(0, linestyle='--', color='gray')

    axs[1].set_xlabel("True Values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_ylim(-b * 0.6, b * 0.6)
    axs[1].set_title(f"{title} - Residual")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"D:\\Articles\\Elmi\\Figs\\{name}.png",
        dpi=1000,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()

# =====================================================
# KNN BEFORE tuning (default)
# =====================================================
knn_before = KNeighborsRegressor()
knn_before.fit(Xtr_sc, ytr)

yprtr_b = knn_before.predict(Xtr_sc)
yprte_b = knn_before.predict(Xte_sc)

plot_model_results(
    ytr, yprtr_b, yte, yprte_b,
    title="KNN (Before Tuning)",
    name="KNN_before_tuning",
    train_color="#74c476",   # سبز
    test_color="#756bb1"    # نارنجی
)

# =====================================================
# KNN AFTER tuning
# =====================================================
knn_after = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    p=1,
    algorithm="auto"
)

knn_after.fit(Xtr_sc, ytr)

yprtr_a = knn_after.predict(Xtr_sc)
yprte_a = knn_after.predict(Xte_sc)

plot_model_results(
    ytr, yprtr_a, yte, yprte_a,
    title="KNN (After Tuning)",
    name="KNN_after_tuning",
    train_color="#74c476",   # سبز
    test_color="#756bb1"     # بنفش
)


# In[21]:


plt.plot(['DT', ' RF',"CAT", "KNN"], [0.91, 0.93, 0.93, 0.79], 'r--', label='Befor applying grid serach')
plt.plot(['DT', ' RF',"CAT", "KNN"], [0.93,0.93, 0.93, 0.90],"-.",label='After applying grid serach')
plt.ylabel('R2 scores')
plt.legend()
# plt.savefig(r"D:\Articles\Elmi\Figs\grid.png", dpi=1000,format='png')
plt.show()


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

# =========================
# Load data
# =========================
df = pd.read_excel(
    r"D:\Articles\Elmi\Final.xlsx",
    sheet_name='Dataset',
    header=0,
    engine='openpyxl'
)

X = df.iloc[:, [0,1,2,3,4,5]].values
y = df.iloc[:, 6].values

# =========================
# Train / Test split
# =========================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=0.7, random_state=7
)

# =========================
# Scaling for KNN only
# =========================
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

# =========================
# DEFINE MODELS HERE
# (Only place you need to edit)
# =========================
models = {
    "Decision Tree": {
        "model": DecisionTreeRegressor(
    random_state=0,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=5,
    max_features='sqrt'
),
        "Xtr": Xtr,
        "Xte": Xte
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=0, 
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              max_depth=None, 
                              max_features='auto', 
                              n_estimators=50),
        "Xtr": Xtr,
        "Xte": Xte
    },
    "CatBoost": {
        "model": CatBoostRegressor(
                depth=4,
                learning_rate=0.05,
                iterations=500,
                random_state=0,
                verbose=0
            ),
        "Xtr": Xtr,
        "Xte": Xte
    },
    "KNN (Scaled)": {
        "model": KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    p=1,
    algorithm="auto"
)
,
        "Xtr": Xtr_sc,
        "Xte": Xte_sc
    }
}

# =========================
# Plot settings
# =========================
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=14)

# =========================
# Function: CDF plot
# =========================
def plot_cdf(ax, y_true, y_pred, title):

    rel_err = np.abs((y_pred - y_true) / y_true) * 100
    rel_err = np.sort(rel_err)
    cdf = np.arange(1, len(rel_err)+1) / len(rel_err)

    ax.plot(rel_err, cdf, color='black', linewidth=2)
    ax.axvline(10, linestyle='--', color='gray', label='10% error')

    ax.set_title(title)
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

# =========================
# Train models & plot CDF
# =========================
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
axs = axs.flatten()

for ax, (name, cfg) in zip(axs, models.items()):
    model = cfg["model"]
    model.fit(cfg["Xtr"], ytr)
    y_pred = model.predict(cfg["Xte"])
    plot_cdf(ax, yte, y_pred, name)

plt.tight_layout()
plt.savefig(
    r"D:\Articles\Elmi\Figs\CDF_Comparison_All_Models.png",
    dpi=1000,
    bbox_inches='tight'
)
plt.show()
plt.close()


# In[30]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import ttest_rel, wilcoxon

# =========================
# Load data
# =========================
df = pd.read_excel(
    r"D:\Articles\Elmi\Final.xlsx",
    sheet_name='Dataset',
    header=0,
    engine='openpyxl'
)

y = df.iloc[:, 6].values
X = df.iloc[:, [0,1,2,3,4,5]].values

Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=0.7, random_state=7
)

# =========================
# Helper function
# =========================
def statistical_test(y_true, y_pred_before, y_pred_after):
    err_before = np.abs(y_true - y_pred_before)
    err_after  = np.abs(y_true - y_pred_after)

    mae_before = err_before.mean()
    mae_after  = err_after.mean()

    t_p = ttest_rel(err_before, err_after).pvalue
    w_p = wilcoxon(err_before, err_after).pvalue

    significant = "Yes" if (t_p < 0.05 and w_p < 0.05) else "No"

    return mae_before, mae_after, t_p, w_p, significant

# =========================
# RESULTS STORAGE
# =========================
results = []

# =====================================================
# 1. Decision Tree
# =====================================================
dt_before = DecisionTreeRegressor(random_state=0)
dt_after  = DecisionTreeRegressor(
    random_state=0,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=5,
    max_features='sqrt'
)
dt_before.fit(Xtr, ytr)
dt_after.fit(Xtr, ytr)

y_before = dt_before.predict(Xte)
y_after  = dt_after.predict(Xte)

results.append(
    ["Decision Tree", *statistical_test(yte, y_before, y_after)]
)

# =====================================================
# 2. Random Forest
# =====================================================
rf_before = RandomForestRegressor(random_state=0)
rf_after  = RandomForestRegressor(random_state=0, 
                              min_samples_leaf=1, 
                              min_samples_split=2,
                              max_depth=None, 
                              max_features='auto', 
                              n_estimators=50)

rf_before.fit(Xtr, ytr)
rf_after.fit(Xtr, ytr)

y_before = rf_before.predict(Xte)
y_after  = rf_after.predict(Xte)

results.append(
    ["Random Forest", *statistical_test(yte, y_before, y_after)]
)

# =====================================================
# 3. KNN (Scaled)
# =====================================================
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

knn_before = KNeighborsRegressor()
knn_after  = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    p=1,
    algorithm="auto"
)

knn_before.fit(Xtr_sc, ytr)
knn_after.fit(Xtr_sc, ytr)

y_before = knn_before.predict(Xte_sc)
y_after  = knn_after.predict(Xte_sc)

results.append(
    ["KNN (Scaled)", *statistical_test(yte, y_before, y_after)]
)

# =====================================================
# 4. CatBoost
# =====================================================
from catboost import CatBoostRegressor

cat_before = CatBoostRegressor(verbose=0, random_state=42)
cat_after  = CatBoostRegressor(
                depth=4,
                learning_rate=0.05,
                iterations=500,
                random_state=0,
                verbose=0
            )

cat_before.fit(Xtr, ytr)
cat_after.fit(Xtr, ytr)

y_before = cat_before.predict(Xte)
y_after  = cat_after.predict(Xte)

results.append(
    ["CatBoost", *statistical_test(yte, y_before, y_after)]
)

# =========================
# Create final table
# =========================
columns = [
    "Model",
    "MAE Before Optimization",
    "MAE After Optimization",
    "Paired t-test p-value",
    "Wilcoxon p-value",
    "Statistical significance (α=0.05)"
]

table6 = pd.DataFrame(results, columns=columns)

print(table6)

# Optional: save
table6.to_excel(
    r"D:\Articles\Elmi\Table6_Statistical_Tests.xlsx",
    index=False
)


# In[27]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from scipy.stats import ttest_rel, wilcoxon

# =========================
# Load data
# =========================
df = pd.read_excel(
    r"D:\Articles\Elmi\Final.xlsx",
    sheet_name='Dataset',
    header=0,
    engine='openpyxl'
)

y = df.iloc[:, 6].values
X = df.iloc[:, [0,1,2,3,4,5]].values

Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=0.7, random_state=7
)

# =========================
# RMSE-based statistical test
# =========================
def statistical_test_rmse(y_true, y_pred_before, y_pred_after, alpha=0.10):
    err_before = (y_true - y_pred_before) ** 2
    err_after  = (y_true - y_pred_after) ** 2

    rmse_before = np.sqrt(err_before.mean())
    rmse_after  = np.sqrt(err_after.mean())

    t_p = ttest_rel(err_before, err_after).pvalue
    w_p = wilcoxon(err_before, err_after).pvalue

    significant = "Yes" if (t_p < alpha and w_p < alpha) else "No"

    return rmse_before, rmse_after, t_p, w_p, significant

# =========================
# RESULTS STORAGE
# =========================
results = []

# =====================================================
# 1. Decision Tree
# =====================================================
dt_before = DecisionTreeRegressor(random_state=0)
dt_after  = DecisionTreeRegressor(
    random_state=0,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=5,
    max_features='sqrt'
)

dt_before.fit(Xtr, ytr)
dt_after.fit(Xtr, ytr)

y_before = dt_before.predict(Xte)
y_after  = dt_after.predict(Xte)

results.append(
    ["Decision Tree", *statistical_test_rmse(yte, y_before, y_after)]
)

# =====================================================
# 2. Random Forest
# =====================================================
rf_before = RandomForestRegressor(random_state=0)
rf_after  = RandomForestRegressor(
    random_state=0,
    n_estimators=50,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features='sqrt'
)

rf_before.fit(Xtr, ytr)
rf_after.fit(Xtr, ytr)

y_before = rf_before.predict(Xte)
y_after  = rf_after.predict(Xte)

results.append(
    ["Random Forest", *statistical_test_rmse(yte, y_before, y_after)]
)

# =====================================================
# 3. KNN (Scaled)
# =====================================================
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

knn_before = KNeighborsRegressor()
knn_after  = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    p=1
)

knn_before.fit(Xtr_sc, ytr)
knn_after.fit(Xtr_sc, ytr)

y_before = knn_before.predict(Xte_sc)
y_after  = knn_after.predict(Xte_sc)

results.append(
    ["KNN (Scaled)", *statistical_test_rmse(yte, y_before, y_after)]
)

# =====================================================
# 4. CatBoost
# =====================================================
cat_before = CatBoostRegressor(verbose=0, random_state=0)
cat_after  = CatBoostRegressor(
    depth=4,
    learning_rate=0.05,
    iterations=500,
    random_state=0,
    verbose=0
)

cat_before.fit(Xtr, ytr)
cat_after.fit(Xtr, ytr)

y_before = cat_before.predict(Xte)
y_after  = cat_after.predict(Xte)

results.append(
    ["CatBoost", *statistical_test_rmse(yte, y_before, y_after)]
)

# =========================
# Create final table
# =========================
columns = [
    "Model",
    "RMSE Before Optimization",
    "RMSE After Optimization",
    "Paired t-test p-value",
    "Wilcoxon p-value",
    "Statistical significance (α=0.10)"
]

table_rmse = pd.DataFrame(results, columns=columns)

print("\n===== Statistical Significance Test Based on RMSE =====")
print(table_rmse)

# Optional: save table
table_rmse.to_excel(
    r"D:\Articles\Elmi\Table_RMSE_Statistical_Tests.xlsx",
    index=False
)


# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor

# =========================
# Load data
# =========================
# df = pd.read_excel("/content/Final.xlsx", sheet_name="Dataset")

X = df.iloc[:, [0,1,2,3,4,5]]
y = df.iloc[:, 6]
feature_names = X.columns.tolist()

Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=0.7, random_state=42
)

# =========================
# Train model
# =========================
model = CatBoostRegressor(
    depth=4,
    learning_rate=0.05,
    iterations=500,
    random_state=42,
    verbose=0
)
model.fit(Xtr, ytr)

# =========================
# Permutation Importance
# =========================
perm = permutation_importance(
    model,
    Xte,
    yte,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

perm_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": perm.importances_mean
}).sort_values("Importance", ascending=True)

# =========================
# Plot
# =========================
plt.figure(figsize=(6, 5))
plt.barh(
    perm_df["Feature"],
    perm_df["Importance"],
    color="#4C72B0",
    edgecolor="black"
)

plt.xlabel("Mean decrease in MAE")
plt.title("Permutation Importance (CatBoost)")
plt.tight_layout()
plt.show()


# In[34]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# بارگذاری داده‌ها از فایل Excel
df = pd.read_excel(
    r"D:\Articles\Elmi\Final.xlsx",
    sheet_name='Dataset',
    header=0,
    engine='openpyxl'
)

# جداسازی ویژگی‌ها و هدف
y = df.iloc[:, 6].to_numpy().reshape((-1, 1)).ravel()  # تبدیل به آرایه یک بعدی
X = df.iloc[:, [0, 1, 2, 3, 4, 5]].to_numpy()

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ایجاد و آموزش مدل CatBoost
model = CatBoostRegressor(
    depth=4,
    learning_rate=0.05,
    iterations=500,
    random_state=0,
    verbose=0
)
model.fit(X_train, y_train)

# بدست آوردن اهمیت ویژگی‌ها
importances = model.get_feature_importance()

# ایجاد DataFrame برای نمایش اهمیت ویژگی‌ها
feature_importance_df = pd.DataFrame({
    'Feature': df.columns[[0, 1, 2, 3, 4, 5]],
    'Importance': importances
})
feature_importance_df = feature_importance_df.sort_values(
    by='Importance',
    ascending=False
)

# نمایش عددی اهمیت ویژگی‌ها
print(feature_importance_df)

# نمایش نمودار اهمیت ویژگی‌ها
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df['Feature'],
    feature_importance_df['Importance']
)
plt.xlabel('Importance')
plt.title('Feature Importance (CatBoost)')
plt.gca().invert_yaxis()
plt.show()


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1) YOUR DATA (EDIT HERE)
# =========================
# اگر X و y را از قبل داری، این بخش را حذف کن
# df = pd.read_excel(r"D:\Articles\Elmi\Final.xlsx", sheet_name="Dataset", engine="openpyxl")
# y = df.iloc[:, 6].values
# X = df.iloc[:, [0,1,2,3,4,5]].values

# اسم فیچرها را مطابق ستون‌ها وارد کن
features = [
    "Feature0", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5"
]

# =========================
# 2) TRAIN/TEST SPLIT (fixed seed)
# =========================
random_seed = 42
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.7, random_state=7)

# =========================
# 3) PUT YOUR MODELS HERE
# =========================
# ---- Decision Tree (before / after) ----
dt_before = DecisionTreeRegressor(random_state=0)  # مثال
dt_after  = DecisionTreeRegressor(
    random_state=0,
    max_depth=11,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features=None
)

# ---- Random Forest (before / after) ----
rf_before = RandomForestRegressor(random_state=0)  # مثال
rf_after  = RandomForestRegressor(
    random_state=0,
    n_estimators=80,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)

# =========================
# 4) FIT
# =========================
dt_before.fit(Xtr, ytr)
dt_after.fit(Xtr, ytr)

rf_before.fit(Xtr, ytr)
rf_after.fit(Xtr, ytr)

# =========================
# 5) HELPERS
# =========================
def summarize_decision_tree(model: DecisionTreeRegressor, name: str):
    tree = model.tree_
    depth = model.get_depth()
    n_leaves = model.get_n_leaves()

    # criterion for regressor usually 'squared_error' (MSE)
    criterion = getattr(model, "criterion", "N/A")

    # pruning info (only meaningful if you used ccp_alpha > 0)
    ccp_alpha = getattr(model, "ccp_alpha", 0.0)

    print(f"\n===== {name} (Decision Tree) =====")
    print(f"Criterion (split): {criterion}  (برای رگرسیون معادل MSE)")
    print(f"Max depth (final): {depth}")
    print(f"Number of leaves : {n_leaves}")
    print(f"ccp_alpha (pruning): {ccp_alpha}")

def save_tree_plot(model: DecisionTreeRegressor, name: str, save_path: str, max_depth_to_show=4):
    plt.figure(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=features,
        filled=True,
        rounded=True,
        max_depth=max_depth_to_show,  # برای خوانایی، عمق نمایش را محدود می‌کنیم
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()

def save_tree_rules(model: DecisionTreeRegressor, name: str, save_path_txt: str, max_depth_rules=6):
    rules = export_text(model, feature_names=features, max_depth=max_depth_rules)
    with open(save_path_txt, "w", encoding="utf-8") as f:
        f.write(f"{name}\n")
        f.write(rules)
    print(f"Rules saved to: {save_path_txt}")

def summarize_random_forest(model: RandomForestRegressor, name: str):
    criterion = getattr(model, "criterion", "N/A")  # usually 'squared_error'
    n_estimators = getattr(model, "n_estimators", len(model.estimators_))

    depths = [estimator.get_depth() for estimator in model.estimators_]
    leaves = [estimator.get_n_leaves() for estimator in model.estimators_]

    print(f"\n===== {name} (Random Forest) =====")
    print(f"Criterion (split): {criterion}  (برای رگرسیون معادل MSE)")
    print(f"n_estimators: {n_estimators}")
    print(f"Depth  (mean/median/max): {np.mean(depths):.2f} / {np.median(depths):.2f} / {np.max(depths)}")
    print(f"Leaves (mean/median/max): {np.mean(leaves):.2f} / {np.median(leaves):.2f} / {np.max(leaves)}")

def save_one_rf_tree_plot(model: RandomForestRegressor, tree_index: int, name: str, save_path: str, max_depth_to_show=4):
    est = model.estimators_[tree_index]
    plt.figure(figsize=(18, 9))
    plot_tree(
        est,
        feature_names=features,
        filled=True,
        rounded=True,
        max_depth=max_depth_to_show,
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()

# =========================
# 6) REPORT + EXPORTS
# =========================
# Decision Tree summaries
summarize_decision_tree(dt_before, "DT Before Tuning")
summarize_decision_tree(dt_after,  "DT After  Tuning")

# Save DT plots (show first levels for readability)
save_tree_plot(dt_after, "DT After Tuning",
               save_path=r"D:\Articles\Elmi\Figs\DT_after_tree.png",
               max_depth_to_show=4)

# Save DT rules (text)
save_tree_rules(dt_after, "DT After Tuning",
                save_path_txt=r"D:\Articles\Elmi\Figs\DT_after_rules.txt",
                max_depth_rules=6)

# Random Forest summaries
summarize_random_forest(rf_before, "RF Before Tuning")
summarize_random_forest(rf_after,  "RF After  Tuning")

# Save ONE representative RF tree (مثلاً درخت شماره 0)
save_one_rf_tree_plot(rf_after, tree_index=0,
                      name="RF After Tuning - Tree 0",
                      save_path=r"D:\Articles\Elmi\Figs\RF_after_tree0.png",
                      max_depth_to_show=4)

print("\nDone.")


# In[ ]:




