import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_ids = test['Id']

y_train = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)

# 合并数据集进行预处理
ntrain = train.shape[0]
all_data = pd.concat([train, test]).reset_index(drop=True)


# 数据预处理函数
def preprocess_data(df):
    # 删除ID列
    df = df.drop('Id', axis=1)

    # 处理缺失值 - 区分不同类型
    # 对于有"无"含义的缺失值，用'None'填充
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in none_cols:
        df[col] = df[col].fillna('None')

    # 对于数值型缺失值，用0填充
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        df[col] = df[col].fillna(0)

    # 对LotFrontage使用邻居分组中位数填充
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

    # 对其他缺失值，用众数或中位数填充
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


# 特征工程函数
def feature_engineering(df):
    # 创建新特征
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GarageArea']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

    # 房屋年龄相关特征
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

    # 质量与面积的交互特征
    df['OverallQual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']

    # 对偏态分布的数值特征进行对数变换
    skewed_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
    for feature in skewed_features:
        if feature in df.columns:
            df[feature] = np.log1p(df[feature])

    return df


# 编码分类变量
def encode_features(df):
    # 对有序分类变量进行标签编码
    qual_dict = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_dict).fillna(0).astype(int)

    # 对其他分类变量使用LabelEncoder
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col not in qual_cols:
            lbl = LabelEncoder()
            lbl.fit(list(df[col].values))
            df[col] = lbl.transform(list(df[col].values))

    return df


# 应用预处理和特征工程
print("进行数据预处理和特征工程...")
all_data = preprocess_data(all_data)
all_data = feature_engineering(all_data)
all_data = encode_features(all_data)

# 分割回训练集和测试集
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# 对目标变量进行对数变换（解决右偏分布）
y_train_log = np.log1p(y_train)


# 定义评估函数
def rmse_cv(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()


# 创建多个基模型
# 1 使用鲁棒缩放器的Lasso回归
lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.0005, random_state=1)
)

# 2 岭回归
ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=0.5, random_state=1)
)

# 3 弹性网络
enet = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3)
)

# 4 梯度提升树
gbr = GradientBoostingRegressor(
    n_estimators=3000, learning_rate=0.05, max_depth=4,
    max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
    loss='huber', random_state=5
)

# 5 XGBoost
model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.5, gamma=0.05, learning_rate=0.05,
    max_depth=3, min_child_weight=1.8, n_estimators=2200,
    reg_alpha=0.5, reg_lambda=0.8, subsample=0.5,
    random_state=7, nthread=-1
)

# 6 LightGBM
model_lgb = lgb.LGBMRegressor(
    objective='regression', num_leaves=5, learning_rate=0.05,
    n_estimators=720, max_bin=55, bagging_fraction=0.8,
    bagging_freq=5, feature_fraction=0.2, feature_fraction_seed=9,
    bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
    verbose=-1
)

# 评估基模型
print("评估基模型性能:")
models = {
    "Lasso": lasso,
    "Ridge": ridge,
    "ElasticNet": enet,
    "Gradient Boosting": gbr,
    "XGBoost": model_xgb,
    "LightGBM": model_lgb
}

for name, model in models.items():
    score = rmse_cv(model, X_train, y_train_log)
    print(f"{name}: {score:.6f}")

# 训练最终模型并生成预测
print("训练最终模型...")

# 训练所有模型
lasso.fit(X_train, y_train_log)
ridge.fit(X_train, y_train_log)
enet.fit(X_train, y_train_log)
gbr.fit(X_train, y_train_log)
model_xgb.fit(X_train, y_train_log)
model_lgb.fit(X_train, y_train_log)

# 生成预测（注意要反向对数变换）
lasso_pred = np.expm1(lasso.predict(X_test))
ridge_pred = np.expm1(ridge.predict(X_test))
enet_pred = np.expm1(enet.predict(X_test))
gbr_pred = np.expm1(gbr.predict(X_test))
xgb_pred = np.expm1(model_xgb.predict(X_test))
lgb_pred = np.expm1(model_lgb.predict(X_test))

# 加权平均集成预测结果
final_pred = (
        0.15 * lasso_pred +
        0.15 * ridge_pred +
        0.10 * enet_pred +
        0.20 * gbr_pred +
        0.20 * xgb_pred +
        0.20 * lgb_pred
)

# 创建提交文件 (保留6位小数)
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': [f"{price:.6f}" for price in final_pred]
})

# 保存结果
submission.to_csv('improved_submission.csv', index=False)
print("改进后的提交文件已保存为 'improved_submission.csv'")
print(f"文件包含 {len(submission)} 行预测结果")
print("前5行预测结果示例:")
print(submission.head())