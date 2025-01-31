# 双色球数据抓取与预测系统

- `data.py`：数据抓取、处理、存储与可视化核心脚本。  
- `time_series_prediction_with_all.py`：基于历史数据的多模型预测与推荐系统。

---

## 功能特性

### 1. `data.py` 功能
- **数据抓取**：支持按期数、期号、日期范围从官网抓取数据。
- **数据处理**：自动去重、排序、生成统计结果（频率、概率）。
- **可视化**：生成热力图、分布图等图表。
- **多线程加速**：使用线程池提升数据抓取效率。
- **Excel集成**：结果保存为带格式的Excel文件，支持高亮显示。
- **日志系统**：详细记录运行状态和错误信息。

### 2. `time_series_prediction_with_all.py` 功能
- **8种预测模型**：  
  - 传统方法：移动平均（MA）、指数平滑（ES）  
  - 机器学习：随机森林（RF）、支持向量机（SVR）、贝叶斯回归（Bayes）  
  - 时间序列：ARIMA、LSTM、混合模型（Hybrid）  
- **智能推荐**：基于聚类算法生成5组综合推荐号码。
- **可视化**：生成红球/蓝球走势图。
- **并发执行**：多线程加速模型预测。

---

## 环境依赖
- Python 3.8+
- 依赖库安装命令：
  ```bash
  pip install pandas numpy matplotlib seaborn requests openpyxl scikit-learn statsmodels tensorflow kmodes
  ```

---

## 使用步骤

### 1. 运行 `data.py`
`python data.py

- **交互式输入**：根据提示选择查询方式（期数、期号、日期）。
- **输出结果**：
  - `outputs/<下一期号>/`：包含Excel统计文件、可视化图表、预测结果。
  - 示例目录结构：
    ```
    outputs/20240123_153000/
    ├── excels/               # 统计结果Excel
    ├── charts/               # 热力图、分布图
    └── predictions/          # 预测详情与推荐号码
    ```

### 2. 运行预测脚本（自动调用）
`data.py` 执行完毕后会自动调用 `time_series_prediction_with_all.py`，生成：
- **走势图**：红球6个位置及蓝球的历史走势。
- **推荐号码**：在 `predictions/results/preview结果.txt` 中查看。

---

## 配置说明
### 1. 设置请求头
在 `data.py` 中修改以下环境变量以绕过反爬机制：
```python
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
COOKIE = "HMF_CI=eb51f625c54193e2c3da483f1f8a20849df1836e651910d5c13682ac94490c626fd7ee9b8958ab34c8abc8a8f7fc5dd52f24fdae80b7aaa45255ee8f42c1015380; 21_vq=3"
```

### 2. 目录结构
- 所有输出文件保存在 `outputs/` 下，按下一期号和时间戳分类。
- 日志文件：`data.log`（主脚本）、`prediction.log`（预测脚本）。

---

## 文件说明
- `data.py`：主脚本，负责数据全流程处理。
- `time_series_prediction_with_all.py`：预测脚本，依赖主脚本生成的Excel文件。
- `outputs/`：程序运行后自动生成的结果目录。

---

## 常见问题（FAQ）

### Q1: 抓取数据时提示“请求被拒绝”
- **原因**：网站反爬机制生效。
- **解决**：更新 `USER_AGENT` 和 `COOKIE`，或添加IP代理。

### Q2: 运行时报错“缺少模块”
- **原因**：依赖未安装完整。
- **解决**：重新执行 `pip install -r requirements.txt`（需自行生成依赖列表）。

### Q3: 预测结果不准确
- **原因**：模型基于历史数据统计，彩票本质为随机事件。
- **建议**：仅作为参考，理性购彩。
