import numpy as np
import pylab as plt
def FrankWolfe(learning_rate, x, y, max_iter=200, tolerance=1e-6):
    #point = np.random.randn(x.shape[1], 1) * 0.001
    point = np.zeros((x.shape[1], 1))
    gaps = []
    losses = []
    current_WX = np.dot(y, x)
    for t in range(max_iter):
        x_a = np.squeeze(np.dot(x, point))
        gradient = np.dot(x.T, x_a) - current_WX
        max_index = np.argmax(np.abs(gradient))
        g_t1 = np.squeeze(np.dot(point.T, gradient))
        g_t2 =gradient[max_index] * np.sign(-gradient[max_index]) * learning_rate
        g_t = g_t1 - g_t2
        gaps.append(g_t)
        if g_t < tolerance:
            break
        Ldt2 = x[:, max_index] * np.sign(-gradient[max_index]) * learning_rate - x_a
        stride = min(np.dot(Ldt2, y - x_a) / np.dot(Ldt2,Ldt2), 1.)
        point = (1. - stride) * point
        point[max_index] = point[max_index] + stride * np.sign(-gradient[max_index]) * learning_rate
    return point, gaps


def display_performance(gaps, title):
    plt.plot(gaps)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('FW gap')
    plt.title(title)
    plt.grid()
    plt.show()
