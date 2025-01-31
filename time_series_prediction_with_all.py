# time_series_prediction_with_all.py
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端

import os
import sys
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge

# statsmodels
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

# tensorflow (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# kmodes for clustering
try:
    from kmodes.kmodes import KModes
except ImportError:
    print("请先安装 kmodes 模块: pip install kmodes")
    sys.exit(1)

from collections import Counter

# OpenPyXL for Excel operations
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.styles import PatternFill

import logging
from pathlib import Path

import argparse  # 导入 argparse
from concurrent.futures import ThreadPoolExecutor, as_completed  # 引入并发模块

# 设置命令行参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description="时间序列预测脚本")
    parser.add_argument("excel_file", help="Excel文件路径")
    parser.add_argument("output_dir", help="输出文件夹路径")
    return parser.parse_args()

args = parse_arguments()

# 设置日志
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = Path(args.output_dir)  # 使用传递的输出目录
LOGS_DIR = OUTPUTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # 设置为 DEBUG 级别以获取详细信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到终端
        logging.FileHandler(LOGS_DIR / "prediction.log", encoding='utf-8')  # 同时输出到日志文件
    ]
)

# 设置中文字体（若需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义输出目录
CHARTS_DIR = OUTPUTS_DIR / "charts"
RED_PLOTS_DIR = CHARTS_DIR / "red_plots"
BLUE_PLOTS_DIR = CHARTS_DIR / "blue_plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DETAILS_DIR = PREDICTIONS_DIR / "details"
RESULTS_DIR = PREDICTIONS_DIR / "results"

# 创建所有必要的目录（如果未创建）
for directory in [RED_PLOTS_DIR, BLUE_PLOTS_DIR, DETAILS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 定义文本文件的路径
details_file_path = DETAILS_DIR / "preview详情.txt"
results_file_path = RESULTS_DIR / "preview结果.txt"

# 定义预测方法参数
DEFAULT_N_LAGS = 5
DEFAULT_EPOCHS = 10

def moving_average_prediction(series, window=5):
    try:
        if len(series) < window:
            return series.iloc[-1]
        return series.iloc[-window:].mean()
    except Exception as e:
        logging.error(f"移动平均预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])

def exponential_smoothing_prediction(series, alpha=0.3):
    try:
        model = SimpleExpSmoothing(series.astype(float)).fit(smoothing_level=alpha, optimized=False)
        forecast = model.forecast(1)
        return forecast.iloc[0]
    except Exception as e:
        logging.error(f"指数平滑预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])

def create_supervised_data(values, n_lags=5):
    X, y = [], []
    for i in range(len(values) - n_lags):
        X.append(values[i:i+n_lags])
        y.append(values[i+n_lags])
    return np.array(X), np.array(y)

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

# def hybrid_prediction(series):
#     try:
#         pred_arima = arima_prediction(series, p=1, d=0, q=1)
#         pred_lstm  = lstm_prediction(series, n_lags=5, epochs=10)
#         return (pred_arima + pred_lstm)/2.0
#     except Exception as e:
#         logging.error(f"混合模型预测失败: {e}", exc_info=True)
#         return float(series.iloc[-1])
def hybrid_prediction(series, weights={'arima': 0.6, 'lstm': 0.4}):
    try:
        pred_arima = arima_prediction(series, p=1, d=0, q=1)
        pred_lstm  = lstm_prediction(series, n_lags=5, epochs=10)
        return weights['arima'] * pred_arima + weights['lstm'] * pred_lstm
    except Exception as e:
        logging.error(f"混合模型预测失败: {e}", exc_info=True)
        return float(series.iloc[-1])


def predict_next_number(series, method="ma"):
    try:
        if method == "ma":
            return moving_average_prediction(series, window=5)
        elif method == "es":
            return exponential_smoothing_prediction(series, alpha=0.3)
        elif method in ["rf", "svr", "bayes"]:
            return ml_prediction(series, method=method, n_lags=5)
        elif method == "arima":
            return arima_prediction(series, p=1, d=0, q=1)
        elif method == "lstm":
            return lstm_prediction(series, n_lags=5, epochs=10)
        elif method == "hybrid":
            return hybrid_prediction(series)
        else:
            raise ValueError(f"Unsupported method: {method}")
    except Exception as e:
        logging.error(f"预测方法 {method} 失败: {e}", exc_info=True)
        return float(series.iloc[-1])

def plot_blue_last10(periods, last10, new_value, new_period, method="ma"):
    try:
        display_periods = [str(p)[-3:] for p in periods]
        display_new_period = str(new_period)[-3:]

        # 输出用于绘图的期号和蓝球号码
        logging.debug(f"绘图蓝球期号: {display_periods}")
        logging.debug(f"绘图新蓝球期号: {display_new_period}")
        logging.debug(f"绘图蓝球号码数据: {last10}")

        # 创建 Figure 和 Axes 对象
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制历史数据
        ax.plot(display_periods, last10, marker='o', color='b', label='历史数据')

        # 绘制预测值
        ax.plot(display_new_period, new_value, marker='D', color='r', label='新预测')

        # 设置标题和标签
        ax.set_title(f"蓝球走势（{method.upper()}）")
        ax.set_xlabel("期号")
        ax.set_ylabel("蓝球号码")
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()

        # 保存图像
        time_str = time.strftime("%Y%m%d_%H%M%S")
        filename = BLUE_PLOTS_DIR / f"蓝球走势_{method.upper()}_{time_str}.png"
        fig.savefig(filename, dpi=100)
        plt.close(fig)

        logging.info(f"蓝球走势图已保存到 {filename}")
    except Exception as e:
        logging.error(f"绘制蓝球走势图失败 ({method}): {e}", exc_info=True)

def plot_red_positions_in_one_figure(df, predictions, method="ma", lookback=30, periods=None):
    try:
        display_periods = [str(p)[-3:] for p in periods]
        display_new_period = str(max(periods) + 1)[-3:]

        # 输出用于绘图的期号和红球号码
        logging.debug(f"绘图红球期号: {display_periods}")
        logging.debug(f"绘图新红球期号: {display_new_period}")
        for pos in range(1, 7):
            col_name = f"红球号码{pos}"
            series_pos = df[col_name].tail(lookback).iloc[-10:]
            logging.debug(f"绘图红球位置 {pos} 数据: {series_pos.tolist()}")

        # 创建 Figure 和 Axes 对象
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = ["blue", "green", "orange", "purple", "brown", "grey"]
        labels = [f"红{pos}历史" for pos in range(1, 7)]

        for pos in range(1, 7):
            col_name = f"红球号码{pos}"
            series_pos = df[col_name].tail(lookback).iloc[-10:]
            last10_pos = series_pos.tolist()

            # 绘制历史数据
            ax.plot(display_periods, last10_pos, marker='o', color=colors[pos-1], label=labels[pos-1] if pos ==1 else "")

            # 绘制预测值
            ax.plot(display_new_period, predictions[pos], marker='D', color=colors[pos-1])

        # 设置标题和标签
        ax.set_title(f"红球 6 个位置走势（{method.upper()}）")
        ax.set_xlabel("期号")
        ax.set_ylabel("红球号码")
        plt.xticks(rotation=45)

        # 创建图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        plt.tight_layout()

        # 保存图像
        time_str = time.strftime("%Y%m%d_%H%M%S")
        filename = RED_PLOTS_DIR / f"红球走势_{method.upper()}_{time_str}.png"
        fig.savefig(filename, dpi=100)
        plt.close(fig)

        logging.info(f"红球走势图已保存到 {filename}")
    except Exception as e:
        logging.error(f"绘制红球走势图失败 ({method}): {e}", exc_info=True)

# def recommend_by_method(df, method="ma", lookback=None):
#     if len(df) < 15:
#         return [f"数据不足15期，{method.upper()} 方法暂不执行。"], None

#     # 使用传递的 lookback 参数，如果未传递，则使用全部数据
#     lookback = len(df) if lookback is None else lookback
#     logging.info(f"方法 {method.upper()} 使用的 lookback 值: {lookback}")

#     # 提取用于预测的蓝球数据
#     blue_series = df["蓝球号码"].tail(lookback)
#     logging.debug(f"蓝球数据 (最近 {lookback} 期):\n{blue_series}")

#     blue_pred_float = predict_next_number(blue_series, method=method)
#     blue_int = max(1, min(16, int(round(blue_pred_float))))

#     last10_blue = list(blue_series.iloc[-10:])
#     last10_periods = list(df["期数"].tail(10))

#     new_period = df.iloc[-1]["期数"] + 1

#     print_buffer = [
#         f"\n========== 使用 {method.upper()} 预测 ==========",
#         f"【{method.upper()}】蓝球最近10期：{last10_blue}",
#         f"【{method.upper()}】蓝球新预测（浮点）: {blue_pred_float:.2f}",
#         f"【{method.upper()}】蓝球取整后: {blue_int}"
#     ]

#     plot_blue_last10(last10_periods, last10_blue, blue_int, new_period, method=method)

#     # 预测红球
#     red_predictions = {}
#     red_vals = []
#     for pos in range(1, 7):
#         col_name = f"红球号码{pos}"
#         series_pos = df[col_name].tail(lookback)
#         logging.debug(f"红球位置 {pos} 数据 (最近 {lookback} 期):\n{series_pos}")

#         pred_val_float = predict_next_number(series_pos, method=method)
#         pred_int = max(1, min(33, int(round(pred_val_float))))

#         last10_pos = list(series_pos.iloc[-10:])
#         print_buffer.append(f"【{method.upper()}】红{pos}最近10期：{last10_pos}")
#         print_buffer.append(
#             f"【{method.upper()}】红{pos}新预测（浮点）: {pred_val_float:.2f}, 取整后: {pred_int}"
#         )

#         red_predictions[pos] = pred_int
#         red_vals.append(pred_int)

#     plot_red_positions_in_one_figure(df, predictions=red_predictions, method=method, lookback=lookback, periods=last10_periods)

#     # 去重+补足+排序
#     reds_unique = sorted(set(red_vals))
#     if len(reds_unique) < 6:
#         remaining = list(set(range(1, 34)) - set(reds_unique))
#         additional = random.sample(remaining, 6 - len(reds_unique))
#         reds_unique.extend(additional)
#         logging.debug(f"红球补充的额外号码: {additional}")
#     reds_unique = sorted(reds_unique[:6])

#     rec_str = f"【{method.upper()}】推荐号码：红球{reds_unique} + 蓝球{blue_int}"
#     print_buffer.append(rec_str)

#     return print_buffer, (reds_unique, blue_int)

#联合预测与智能补充
def recommend_by_method(df, method="ma", lookback=None):
    if len(df) < 15:
        return [f"数据不足15期，{method.upper()} 方法暂不执行。"], None

    lookback = len(df) if lookback is None else lookback
    logging.info(f"方法 {method.upper()} 使用的 lookback 值: {lookback}")

    blue_series = df["蓝球号码"].tail(lookback)
    blue_pred_float = predict_next_number(blue_series, method=method)
    blue_int = max(1, min(16, int(round(blue_pred_float))))

    last10_blue = list(blue_series.iloc[-10:])
    last10_periods = list(df["期数"].tail(10))
    new_period = df.iloc[-1]["期数"] + 1

    print_buffer = [
        f"\n========== 使用 {method.upper()} 预测 ==========",
        f"【{method.upper()}】蓝球最近10期：{last10_blue}",
        f"【{method.upper()}】蓝球新预测（浮点）: {blue_pred_float:.2f}",
        f"【{method.upper()}】蓝球取整后: {blue_int}"
    ]

    plot_blue_last10(last10_periods, last10_blue, blue_int, new_period, method=method)

    red_predictions = {}
    red_vals = []
    for pos in range(1, 7):
        col_name = f"红球号码{pos}"
        series_pos = df[col_name].tail(lookback)
        pred_val_float = predict_next_number(series_pos, method=method)
        pred_int = max(1, min(33, int(round(pred_val_float))))

        last10_pos = list(series_pos.iloc[-10:])
        print_buffer.append(f"【{method.upper()}】红{pos}最近10期：{last10_pos}")
        print_buffer.append(
            f"【{method.upper()}】红{pos}新预测（浮点）: {pred_val_float:.2f}, 取整后: {pred_int}"
        )

        red_predictions[pos] = pred_int
        red_vals.append(pred_int)

    plot_red_positions_in_one_figure(df, predictions=red_predictions, method=method, lookback=lookback, periods=last10_periods)

    # 去重并基于频率补充
    reds_unique = sorted(set(red_vals))
    if len(reds_unique) < 6:
        # 计算剩余号码的出现频率
        red_frequency = df.iloc[:, 3:9].apply(pd.Series.value_counts).fillna(0).astype(int).sum(axis=1)
        remaining = list(set(range(1, 34)) - set(reds_unique))
        remaining_freq = red_frequency[remaining]
        # 根据频率选择补充号码，优先选择出现频率高的号码
        remaining_sorted = remaining_freq.sort_values(ascending=False).index.tolist()
        additional = remaining_sorted[:6 - len(reds_unique)]
        reds_unique.extend(additional)
        logging.debug(f"基于频率补充的红球号码: {additional}")

    reds_unique = sorted(reds_unique[:6])

    rec_str = f"【{method.upper()}】推荐号码：红球{reds_unique} + 蓝球{blue_int}"
    print_buffer.append(rec_str)

    return print_buffer, (reds_unique, blue_int)


def generate_recommendations(results_map, num_recommendations=5):
    red_sets = []
    blue_counts = []
    for method, (buf, res) in results_map.items():
        if res is not None:
            reds, blue = res
            red_sets.append(reds)
            blue_counts.append(blue)

    if not red_sets:
        logging.warning("没有可用的红球预测数据。")
        return []

    def set_to_binary_vector(s, n=33):
        vec = [0]*n
        for num in s:
            vec[num-1] = 1
        return vec

    binary_vectors = [set_to_binary_vector(s) for s in red_sets]
    binary_vectors = np.array(binary_vectors)

    try:
        km = KModes(n_clusters=num_recommendations, init='Huang', n_init=5, verbose=0)
        clusters = km.fit_predict(binary_vectors)
    except Exception as e:
        logging.error(f"K-Modes 聚类失败: {e}", exc_info=True)
        return []

    recommendations = []
    for cluster_num in range(num_recommendations):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_num]
        if not cluster_indices:
            continue
        cluster_reds = [tuple(sorted(red_sets[i])) for i in cluster_indices]
        all_red_numbers = [num for sublist in cluster_reds for num in sublist]
        red_counter = Counter(all_red_numbers)
        representative_reds = sorted([num for num, cnt in red_counter.most_common(6)])

        cluster_blues = [blue_counts[i] for i in cluster_indices]
        blue_counter = Counter(cluster_blues)
        most_common_blue = blue_counter.most_common(1)[0][0]

        recommendations.append((representative_reds, most_common_blue))

    while len(recommendations) < num_recommendations:
        all_red_numbers = [num for s in red_sets for num in s]
        red_counter = Counter(all_red_numbers)
        top_reds = sorted([num for num, cnt in red_counter.most_common(6)])

        blue_counter = Counter(blue_counts)
        top_blue = blue_counter.most_common(1)[0][0]

        if (top_reds, top_blue) not in recommendations:
            recommendations.append((top_reds, top_blue))
        else:
            break

    return recommendations

def run_time_series_prediction(excel_file):
    sheet_name = "开奖结果"
    time_str = time.strftime("%Y年%m月%d日_%H:%M:%S")
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    except FileNotFoundError:
        logging.error(f"错误: 文件 '{excel_file}' 未找到。")
        return
    except ValueError:
        logging.error(f"错误: 工作表 '{sheet_name}' 不存在于 '{excel_file}' 中。")
        return

    # 按期数排序并去重
    df.sort_values(by="期数", ascending=True, inplace=True)
    initial_count = len(df)
    df.drop_duplicates(subset="期数", inplace=True)
    final_count = len(df)
    if initial_count != final_count:
        logging.warning(f"移除了 {initial_count - final_count} 个重复的期数。")
    df.reset_index(drop=True, inplace=True)
    logging.info("读取并整理数据，按期数升序排序并移除重复。")

    logging.info(f"读取完毕，共 {len(df)} 条记录。")
    latest_period = df["期数"].max()
    next_period = latest_period + 1
    logging.info(f"最新一期的期数是：{latest_period}")

    methods = ["ma", "es", "rf", "svr", "bayes", "arima", "lstm", "hybrid"]
    results_map = {}
    next_period = latest_period + 1

    def run_method(m):
        logging.info(f"开始使用方法 {m.upper()} 进行Preview，使用最近 {len(df)} 期数据。")
        buf, res = recommend_by_method(df, method=m, lookback=len(df))  # 使用全部数据
        results_map[m] = (buf, res)
        logging.info(f"方法 {m.upper()} 预测完成。")

    # 使用线程池并行运行预测方法
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_method, m) for m in methods]
        for future in as_completed(futures):
            pass  # 所有结果已存入 results_map

    # 写入Preview详情
    with open(details_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{next_period}期Preview详情-{time_str}\n")
        for m in methods:
            buf, res = results_map[m]
            if buf is not None:
                for line in buf:
                    print(line)
                    f.write(line + '\n')

    # 生成推荐号码
    recommendations = generate_recommendations(results_map, num_recommendations=5)

    # 写入Preview结果
    with open(results_file_path, 'a', encoding='utf-8') as file:
        file.write(f"\n{next_period}期Preview {time_str}\n")
        file.write(f"使用数据：最近 {len(df)} 期\n")
        for m in methods:
            _, res = results_map[m]
            if res is not None:
                reds, blue = res
                print(f"{m.upper()} -> 红球{reds} + 蓝球{blue}")
                file.write(f"{m.upper()} -> 红球{reds} + 蓝球{blue}\n")

        print("\n>>> [Preview] 五组综合<<<\n")
        file.write("\n[Preview] 五组综合\n")
        for idx, (reds, blue) in enumerate(recommendations, 1):
            print(f"Preview {idx}: 红球{reds} + 蓝球{blue}")
            file.write(f"Preview {idx}: 红球{reds} + 蓝球{blue}\n")

        # 综合统计
        red_counts = {i: 0 for i in range(1, 34)}
        blue_counts = {i: 0 for i in range(1, 17)}

        for m in methods:
            _, res = results_map[m]
            if res is not None:
                reds, blue = res
                for r in reds:
                    red_counts[r] += 1
                blue_counts[blue] += 1

        sorted_red = sorted(red_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_blue = sorted(blue_counts.items(), key=lambda x: x[1], reverse=True)

        print("\n>>> [综合统计] 各方法红球号码出现次数Top6 <<<\n")
        file.write("\n[综合统计] 各方法红球号码出现次数Top6\n")
        for i in range(6):
            if i < len(sorted_red):
                num, cnt = sorted_red[i]
                print(f"红球 {num} 出现 {cnt} 次")
                file.write(f"红球 {num} 出现 {cnt} 次\n")

        print("\n>>> [综合统计] 各方法蓝球号码出现次数Top3 <<<\n")
        file.write("\n[综合统计] 各方法蓝球号码出现次数Top3\n")
        for i in range(3):
            if i < len(sorted_blue):
                num, cnt = sorted_blue[i]
                print(f"蓝球 {num} 出现 {cnt} 次")
                file.write(f"蓝球 {num} 出现 {cnt} 次\n")
        # 找出没有被任何方法选中的红球号码
        unselected_reds = [num for num, cnt in red_counts.items() if cnt == 0]
        if unselected_reds:
            print("\n>>> [综合统计] 未被任何方法选中的红球号码 <<<\n")
            file.write("\n[综合统计] 未被任何方法选中的红球号码\n")
            for r in unselected_reds:
                print(f"红球 {r} 未被选中")
                file.write(f"红球 {r} 未被选中\n")

        print("\n>>> [推荐号码] 生成完毕 <<<\n")

def main(excel_file):
    run_time_series_prediction(excel_file)

if __name__ == "__main__":
    main(args.excel_file)
