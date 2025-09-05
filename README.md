# Multi-Model-Ensemble-Prediction-of-housing-prices-0.11961-
kaggle上房价竞赛的多模型集成预测（0.11961）/This is a complete solution（score:0.11961） for participating in the Kaggle "House Prices: Advanced Regression Techniques" competition. The code uses multiple machine learning techniques to predict the selling price of houses in Ames, Iowa, with an evaluation metric of RMSLE (Root Mean Squared Logarithmic Error)

  1.项目概述
   
这是一个参与Kaggle"House Prices: Advanced Regression Techniques"竞赛的完整解决方案。代码使用了多种机器学习技术来预测爱荷华州埃姆斯市的房屋销售价格，评估指标为RMSLE（Root Mean Squared Logarithmic Error）

  2.数据预处理 (preprocess_data函数)

这个方案在数据预处理上主要进行了缺失值处理​​

  对于有"无"含义的特征（如游泳池质量、围栏类型等），用'None'填充
  对于数值型特征（如车库面积、地下室面积等），用0填充
  对LotFrontage使用邻居分组中位数填充
  其他缺失值根据数据类型用众数或中位数填充

  3.特征工程 (feature_engineering函数)

创建新特征

  TotalSF: 总建筑面积（地下室+一层+二层）
  TotalArea: 总面积（包括车库）
  TotalBath: 总浴室数（全浴室+0.5*半浴室）
  TotalPorch: 总门廊面积
  房屋年龄相关特征（HouseAge, RemodAge, IsRemodeled）
  质量与面积的交互特征

偏态处理​​

  对偏态分布的数值特征进行对数变换

4.分类变量编码 (encode_features函数)

​​有序分类变量​​：使用映射编码（如质量评级从'Po'到'Ex'映射为1-5）

​​无序分类变量​​：使用LabelEncoder进行编码


5.模型训练与选择

代码使用了多种模型进行训练和比较

   线性模型​​：Lasso、Ridge、ElasticNet（使用正则化防止过拟合）

   树模型​​：Gradient Boosting、XGBoost、LightGBM

   评估方法​​：5折交叉验证计算RMSE

6.集成预测

   使用加权平均集成多个模型的预测结果，这是提高预测准确性的有效策略

  final_pred = (0.15 * lasso_pred + 0.15 * ridge_pred + 0.10 * enet_pred + 0.20 * gbr_pred + 0.20 * xgb_pred + 0.20 * lgb_pred)
  
