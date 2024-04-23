import csv
import os

import bezier
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


# 使用贝塞尔（Bernstein basis）曲线进行拟合
class BezierCurve:
    def __init__(self, control_points, ):
        self.control_points = control_points
        self.n = len(control_points) - 1

    def bernstein_basis(self, i, n, t):
        """Calculate the i-th Bernstein basis polynomial of degree n at t."""
        return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def evaluate(self, t):
        """Evaluate the Bezier curve at the given parameter t."""
        point = np.zeros_like(self.control_points[0])
        for i in range(self.n + 1):
            point += self.control_points[i] * self.bernstein_basis(i, self.n, t)
        return point

    def fit(self, samples):
        """Fit the Bezier curve to the given samples using the least squares method."""
        t = np.linspace(0, 1, self.n + 1)
        A = np.zeros((len(samples), self.n + 1))
        for i, sample in enumerate(samples):
            for j in range(self.n + 1):
                A[i, j] = self.bernstein_basis(j, self.n, t[i])
        b = samples
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self.control_points = x


def sin_curve(x):
    return np.sin(x * 2 * np.pi)


def sample_curve(n, mu=0, sigma=0.1):
    x = np.linspace(0, 1, n)
    y = sin_curve(x)

    noise = np.random.normal(mu, sigma, n)
    y_with_noise = y + noise

    return x, y_with_noise


# para
line_w = 5
ss = 150
label_s = 35
tick_s = 25
tick_l = 15
spine_w = 3

n_list = [10, 15, 100]  # N

for n_points in n_list:
    degree = n_points - 1
    x_fit = np.linspace(0, 1, 100)
    # 采样点
    x_samples, y_samples = sample_curve(n_points, mu=0, sigma=0.2)

    # 拟合
    curve = bezier.Curve(np.array([x_samples, y_samples]), degree=degree)
    y_fit = curve.evaluate_multi(x_fit)

    # 画图
    fig = plt.figure(figsize=(8, 6))
    rect = [0.2, 0.2, 0.75, 0.75]
    ax = plt.axes(rect)

    plt.plot(x_fit, sin_curve(x_fit), color='green', linewidth=line_w)
    plt.scatter(x_samples, y_samples, color='blue', marker='o', facecolors='none',
                linewidth=spine_w, s=ss, label=f'N={n_points}')
    plt.plot(y_fit[0], y_fit[1], color='red', label=f'M={degree}', linewidth=line_w)
    # 添加文字
    text = 'Bezier'
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

    plt.savefig(f'{save_path}/Bezier_N={n_points}_M={degree}.png', dpi=300)

