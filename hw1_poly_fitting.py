import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.special import comb
from scipy.optimize import curve_fit

# 保存路径
save_path = './hw1'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 随机种子
np.random.seed(42)


def sin_curve(x):
    return np.sin(x * 2 * np.pi)


def sample_curve(n, mu=0, sigma=0.1):
    x = np.linspace(0, 1, n)
    y = sin_curve(x)

    noise = np.random.normal(mu, sigma, n)
    y_with_noise = y + noise

    return x, y_with_noise


# 使用多项式拟合
def polynomial_fit(x, y, degree, alpha=None):
    coeffs = np.polyfit(x, y, degree, rcond=alpha)
    return coeffs


def polynomial_curve(x, coeffs):
    return np.polyval(coeffs, x)


# para
line_w = 5
ss = 150
label_s = 35
tick_s = 25
tick_l = 15
spine_w = 3

n_list = [10, 15, 100]  # N
de_list = [3, 9]  # M
alpha = None  # 正则化参数

poly = []
for degree in de_list:
    for n_points in n_list:
        # 采样点
        x_samples, y_samples = sample_curve(n_points, mu=0, sigma=0.2)

        # 多项式拟合
        x_fit = np.linspace(0, 1, 100)
        coeff = polynomial_fit(x_samples, y_samples, degree, alpha=alpha)
        poly.append(coeff)

        # 画图
        fig = plt.figure(figsize=(8, 6))
        rect = [0.2, 0.2, 0.75, 0.75]
        ax = plt.axes(rect)

        if n_points == 10 and degree == 3:
            plt.plot(x_fit, sin_curve(x_fit), color='green', label='y = sin(x*2*pi)', linewidth=line_w)
            plt.scatter(x_samples, y_samples, color='blue', marker='o', facecolors='none',
                        linewidth=spine_w, s=ss, label='Sampled Points(N=10)')
            plt.plot(x_fit, polynomial_curve(x_fit, coeff), color='red',
                     label='Fitted Curve(M=3)', linewidth=line_w)
        else:
            plt.plot(x_fit, sin_curve(x_fit), color='green', linewidth=line_w)
            plt.scatter(x_samples, y_samples, color='blue', marker='o', facecolors='none',
                        linewidth=spine_w, s=ss, label=f'N={n_points}')
            plt.plot(x_fit, polynomial_curve(x_fit, coeff), color='red', label=f'M={degree}', linewidth=line_w)
            if alpha is not None:
                # 添加文字
                text = 'lnλ = -3'
                plt.text(0.05, 0.05, text, transform=plt.gca().transAxes, fontsize=tick_s)

        plt.xlabel('x', fontsize=label_s)
        plt.ylabel('y', fontsize=label_s)
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.xticks(fontsize=tick_s)
        plt.yticks(fontsize=tick_s)
        # 设置刻度格式
        formatter = ticker.FormatStrFormatter('%.1f')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        plt.tick_params(axis='x', width=spine_w, length=tick_l)
        plt.tick_params(axis='y', width=spine_w, length=tick_l)

        plt.legend(fontsize=16)
        # 设置边框的样式和宽度
        for spine in ax.spines.values():
            spine.set_linewidth(spine_w)  # 设置边框宽度为2个单位

        if alpha is not None:
            plt.savefig(f'{save_path}/N={n_points}_M={degree}_lnl={-3}.png', dpi=300)
        else:
            plt.savefig(f'{save_path}/N={n_points}_M={degree}.png', dpi=300)

# 保存多项式到csv文件
with open(f'{save_path}/poly_regular.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(poly)
