# 1. 移动平均预测（Moving Average Prediction, MA）

## 工作原理
移动平均法是一种简单的时间序列预测方法，通过计算最近 `window` 期的数据的平均值来预测下一期的值。它假设未来的值会接近过去一段时间的平均水平。

## 实现代码
```python
def moving_average_prediction(series, window=5):
    try:
        if len(series) < window:
            return series.iloc[-1]
        return series.iloc[-window:].mean()
    except Exception as e:
        logging.error(f"移动平均预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

## 可以改善的方面
- **窗口大小优化**：当前窗口大小固定为 5，可以通过交叉验证或其他优化方法选择最优窗口大小，以提高预测精度。
- **加权移动平均**：为不同时间点的数据分配不同的权重，例如给最近的数据更高的权重，增强模型对近期趋势的敏感性。
- **多重移动平均**：结合不同窗口大小的移动平均结果，综合考虑短期和长期趋势。

## 可修改的参数
- `window`：窗口大小，决定了用于计算平均值的数据期数。可以通过实验选择最适合的数据窗口。

```python
predicted_value = moving_average_prediction(series, window=7)
```

# 2. 指数平滑预测（Exponential Smoothing Prediction, ES）

## 工作原理
指数平滑法通过对历史数据赋予指数衰减的权重，给予近期数据更高的权重，从而预测下一期的值。简单指数平滑（SES）是一种基础的指数平滑方法，适用于无趋势和季节性的时间序列。

## 实现代码
```python
def exponential_smoothing_prediction(series, alpha=0.3):
    try:
        model = SimpleExpSmoothing(series.astype(float)).fit(smoothing_level=alpha, optimized=False)
        forecast = model.forecast(1)
        return forecast.iloc[0]
    except Exception as e:
        logging.error(f"指数平滑预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

## 可以改善的方面
- **自动优化平滑参数**：当前 `alpha` 固定为 0.3，可以设置 `optimized=True` 让模型自动选择最佳的平滑参数。
- **双重或三重指数平滑**：对于具有趋势或季节性的时间序列，可以采用霍尔特线性趋势模型（双重指数平滑）或霍尔特-温特斯季节性模型（三重指数平滑）。
- **混合模型**：结合指数平滑与其他方法，提高预测的稳定性和准确性。

## 可修改的参数
- `alpha`：平滑系数，决定了权重的衰减速度。值越接近 1，模型越重视近期数据；值越接近 0，模型越重视长期数据。

```python
predicted_value = exponential_smoothing_prediction(series, alpha=0.5)
```

# 3. 机器学习预测（Machine Learning Prediction）

包括以下三种方法：

## 3.1 随机森林回归（Random Forest Regressor, RF）

### 工作原理
随机森林是一种集成学习方法，通过构建多个决策树并对它们的预测结果进行平均，来提高预测的准确性和稳定性。它能够捕捉数据中的非线性关系和复杂模式。

### 实现代码
```python
def ml_prediction(series, method="rf", n_lags=5):
    try:
        s = series.values if hasattr(series, 'values') else np.array(series)
        X, Y = create_supervised_data(s, n_lags)
        if len(X) < 1:
            return float(s[-1])

        if method == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == "svr":
            model = SVR()
        elif method == "bayes":
            model = BayesianRidge()
        else:
            raise ValueError("Unknown ML method in ml_prediction()")

        model.fit(X, Y)
        last_data = s[-n_lags:]
        predicted = model.predict([last_data])
        return float(predicted[0])
    except Exception as e:
        logging.error(f"机器学习预测失败 ({method}): {e}", exc_info=True)
        return float(series.iloc[-1])
```

### 可以改善的方面
- **特征工程**：除了使用滞后值外，可以引入其他特征，如滚动统计量、时间特征（周几、节假日等）来增强模型的预测能力。
- **超参数调优**：使用网格搜索（Grid Search）或随机搜索（Random Search）优化模型的超参数，如 `n_estimators`、`max_depth` 等，以提高模型性能。
- **模型集成**：结合多种机器学习模型的预测结果，利用加权平均或堆叠等方法，进一步提升预测准确性。
- **交叉验证**：采用交叉验证方法评估模型性能，避免过拟合，并选择最佳模型配置。

### 可修改的参数
- `n_lags`：滞后期数，决定了用于预测的历史数据长度。调整此参数以捕捉不同时间范围内的模式。

```python
predicted_value = ml_prediction(series, method="rf", n_lags=7)
```

- `method`：选择使用的机器学习模型，支持 `"rf"`（随机森林）、`"svr"`（支持向量回归）、`"bayes"`（贝叶斯岭回归）。

```python
predicted_value_rf = ml_prediction(series, method="rf")
predicted_value_svr = ml_prediction(series, method="svr")
predicted_value_bayes = ml_prediction(series, method="bayes")
```

### 随机森林特定参数：
- `n_estimators`：决策树的数量，增加树的数量通常会提高性能，但也会增加计算成本。
- `max_depth`：每棵树的最大深度，控制模型的复杂度。

```python
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
```

## 3.2 支持向量回归（Support Vector Regression, SVR）

### 工作原理
SVR 是一种基于支持向量机（SVM）的回归方法，通过寻找一个尽可能平滑的函数来拟合数据，同时允许一定的误差。它在处理高维数据和非线性关系方面表现良好。

### 可以改善的方面
- **核函数选择**：尝试不同的核函数（如线性核、多项式核、径向基核等）以适应数据的不同模式。
- **超参数调优**：优化 `C`、`epsilon`、`gamma` 等超参数，提升模型的预测能力。
- **特征缩放**：对数据进行标准化或归一化处理，以提高 SVR 的性能，特别是当使用径向基核时。

### 可修改的参数
- `kernel`：核函数类型，如 `"linear"`、`"poly"`、`"rbf"`、`"sigmoid"`。

```python
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
```

- `C`：惩罚参数，控制模型对误差的容忍度。较大的 `C` 值会使模型更关注训练数据的准确性。
- `epsilon`：在回归中，`epsilon` 决定了多少误差是可以接受的。
- `gamma`：核函数的系数，影响数据点对模型的影响范围。

## 3.3 贝叶斯岭回归（Bayesian Ridge Regression, Bayes）

### 工作原理
贝叶斯岭回归是一种线性回归方法，通过贝叶斯推断框架引入先验分布，自动调整模型的复杂度，防止过拟合。它对模型参数的不确定性进行建模，适用于高维数据和多重共线性问题。

### 可以改善的方面
- **先验分布选择**：调整先验分布的参数，以适应不同的数据特性。
- **特征选择**：结合特征选择方法，去除不相关或冗余的特征，提升模型性能。
- **结合其他模型**：与其他回归模型结合，形成更强的预测系统。

### 可修改的参数
- `alpha_1` 和 `alpha_2`：贝塔分布的参数，控制模型参数的正则化强度。

```python
model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6)
```

- `lambda_1` 和 `lambda_2`：控制模型参数的精确度。

```python
model = BayesianRidge(lambda_1=1e-6, lambda_2=1e-6)
```

- `n_iter`：迭代次数，决定了贝叶斯更新的步数。

# 4. ARIMA 预测（Autoregressive Integrated Moving Average Prediction, ARIMA）

## 工作原理
ARIMA 模型结合了自回归（AR）、差分（I）和移动平均（MA）三个部分，适用于非季节性的时间序列数据。它通过建模数据的自身过去值和误差，预测未来的值。

## 实现代码
```python
def arima_prediction(series, p=1, d=0, q=1):
    try:
        s = series.values if hasattr(series, 'values') else np.array(series)
        if len(s) < max(p, d, q) + 1:
            return float(s[-1])

        model = ARIMA(s, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast[0])
    except Exception as e:
        logging.error(f"ARIMA预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

## 可以改善的方面
- **参数选择优化**：通过 AIC/BIC 指标或自动化方法（如 `auto_arima`）选择最佳的 `(p, d, q)` 参数组合。
- **季节性调整**：如果数据具有季节性，可以使用 SARIMA（季节性 ARIMA）模型。
- **差分次数调整**：确保数据的平稳性，适当调整差分次数 `d`。
- **模型诊断**：在拟合模型后，进行残差分析，确保模型的假设得到满足。

## 可修改的参数
- `p`：自回归项的数量，表示当前值与前 `p` 个时刻的值相关。
- `d`：差分阶数，用于使时间序列平稳。
- `q`：移动平均项的数量，表示当前值与前 `q` 个误差项相关。

```python
predicted_value = arima_prediction(series, p=2, d=1, q=2)
```

### 季节性参数（如果扩展为 SARIMA）：
- `P`、`D`、`Q`：季节性自回归、差分和移动平均项。
- `s`：季节周期长度。

# 5. 长短期记忆网络预测（Long Short-Term Memory Prediction, LSTM）

## 工作原理
LSTM 是一种特殊的循环神经网络（RNN），能够捕捉时间序列中的长期依赖关系。它通过引入门控机制（输入门、遗忘门、输出门）来控制信息的流动，解决了传统 RNN 的梯度消失问题。

## 实现代码
```python
def lstm_prediction(series, n_lags=5, epochs=10):
    try:
        s = series.values if hasattr(series, 'values') else np.array(series)
        X, Y = create_supervised_data(s, n_lags)
        if len(X) < 1:
            return float(s[-1])

        X = X.reshape(X.shape[0], X.shape[1], 1)
        Y = Y.reshape(-1, 1)

        model = Sequential()
        model.add(LSTM(units=20, input_shape=(n_lags, 1)))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, Y, epochs=epochs, batch_size=1, verbose=0)

        last_data = s[-n_lags:].reshape((1, n_lags, 1))
        pred = model.predict(last_data)
        return float(pred[0][0])
    except Exception as e:
        logging.error(f"LSTM预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

## 可以改善的方面
- **模型复杂度调整**：调整 LSTM 网络的层数和每层的单元数，找到最佳的模型架构。
- **训练参数优化**：调整学习率、批大小（batch size）、优化器等超参数，以提高训练效果。
- **正则化技术**：引入 Dropout 层或 L1/L2 正则化，防止模型过拟合。
- **数据预处理**：对数据进行标准化或归一化处理，提升模型训练的稳定性和收敛速度。
- **早停（Early Stopping）**：引入早停机制，根据验证集的表现提前终止训练，防止过拟合。
- **序列长度调整**：实验不同的滞后期数（`n_lags`），捕捉不同时间范围内的模式。

## 可修改的参数
- `n_lags`：滞后期数，决定了用于预测的历史数据长度。

```python
predicted_value = lstm_prediction(series, n_lags=7, epochs=20)
```

- `epochs`：训练的轮数，更多的轮数可能提升模型性能，但也增加计算时间和过拟合风险。

```python
predicted_value = lstm_prediction(series, n_lags=5, epochs=50)
```

### LSTM 特定参数：
- `units`：LSTM 层中的神经元数量，影响模型的学习能力。
- `optimizer`：优化器类型，如 `'adam'`、`'rmsprop'` 等。
- `batch_size`：批大小，决定了每次参数更新的样本数量。

```python
model = Sequential()
model.add(LSTM(units=50, input_shape=(n_lags, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

# 6. 混合预测（Hybrid Prediction）

## 工作原理
混合预测方法结合了多种预测模型的结果，以期望通过集成不同模型的优势，提升整体的预测准确性和稳定性。在本实现中，混合预测通过平均 ARIMA 和 LSTM 的预测结果来生成最终预测值。

## 实现代码
```python
def hybrid_prediction(series):
    try:
        pred_arima = arima_prediction(series, p=1, d=0, q=1)
        pred_lstm  = lstm_prediction(series, n_lags=5, epochs=10)
        return (pred_arima + pred_lstm)/2.0
    except Exception as e:
        logging.error(f"混合模型预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

## 可以改善的方面
- **多模型集成**：不仅仅结合 ARIMA 和 LSTM，还可以引入其他模型（如 RF、SVR、贝叶斯岭回归）进行集成，采用加权平均或投票机制来决定最终预测值。
- **动态权重调整**：根据模型在验证集上的表现，为不同模型分配不同的权重，而不是简单的平均。
- **堆叠模型（Stacking）**：使用一个元模型（如线性回归）来学习如何最优地结合各个基础模型的预测结果。
- **模型选择机制**：根据时间序列的特性（如趋势性、季节性）动态选择适合的模型组合进行混合预测。

## 可修改的参数
- **集成策略**：当前使用的是简单的平均，可以修改为加权平均或其他集成方法。

```python
def hybrid_prediction(series, weights={'arima': 0.6, 'lstm': 0.4}):
    try:
        pred_arima = arima_prediction(series, p=1, d=0, q=1)
        pred_lstm  = lstm_prediction(series, n_lags=5, epochs=10)
        return weights['arima'] * pred_arima + weights['lstm'] * pred_lstm
    except Exception as e:
        logging.error(f"混合模型预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])
```

- **模型数量和类型**：可以调整参与混合的模型数量和类型，根据实际情况选择最适合的组合。

# 7. 预测方法选择与推荐

## 工作原理
在脚本中，通过调用不同的预测方法，生成多组预测结果，并通过聚类算法（K-Modes）将这些结果进行聚类，最终生成综合的推荐号码。这种方法旨在综合多个预测方法的优势，提供更为稳健的预测结果。

## 可以改善的方面
- **聚类算法优化**：尝试其他聚类算法（如 K-Means、DBSCAN）或调整 K-Modes 的参数（如聚类数量、初始化方法）以提高聚类质量。
- **多样化预测方法**：引入更多不同类型的预测方法，增加预测结果的多样性，提升聚类的效果。
- **后处理策略**：在聚类后对推荐号码进行进一步的筛选和优化，如去除重复号码、确保号码范围覆盖等。
- **权重机制**：根据各预测方法的历史表现，为不同方法的预测结果分配不同的权重，提升推荐的准确性。

## 可修改的参数
- `num_recommendations`：推荐号码的组数，可以根据需要调整。

```python
recommendations = generate_recommendations(results_map, num_recommendations=3)
```

### K-Modes 特定参数：
- `n_clusters`：聚类的数量，对应推荐号码的组数。
- `init`：初始化方法，如 `'Huang'` 或 `'Cao'`。
- `n_init`：初始化方法的运行次数，选择最佳聚类结果。

```python
km = KModes(n_clusters=5, init='Cao', n_init=10, verbose=0)
```

### 推荐号码生成逻辑：可以修改生成推荐号码的逻辑，如选择最频繁出现的号码、结合热度等策略。

```python
def generate_recommendations(results_map, num_recommendations=5, top_n=3):
    # 修改为选择每个簇中最频繁出现的前 top_n 个号码
    ...
```
