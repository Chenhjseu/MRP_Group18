import numpy as np
import pylab as plt
import random


# block decent - for Frank wolfre
def FW_BD(x, y, point = None, max_iter=200, tolerance=1e-100, partition = 3):
    if point is None:
        #point = np.random.randn(x.shape[1], 1) * 1000
        point = np.array([233., 1924., 182., -1191., -575., 643., 205., 385., 1028., 199.]).T
    point = point.reshape(x.shape[1], 1)
    blocks_index = get_blocks(x,partition = partition)
    #blocks_index = [(0, 1), (1, 3), (3, 10)]
    gaps = []
    loses = []
    for t in range(max_iter):
        gap, vk = FWSC_state(x, y, point)
        gaps.append(gap)
        loses.append(loss_function(x, y, point))
        if abs(gap) < tolerance:
            break
        random_index = random.randint(0, len(blocks_index)-1)
        b_index = blocks_index[random_index]
        splited_vector = point[b_index[0]:b_index[1]]
        t_point_,  t_gaps, t_loses = FrankWolfewithSC(x[:,b_index[0]:b_index[1]], y, point = splited_vector)
        point[b_index[0]:b_index[1]] = t_point_
    return point, gaps, loses


#adaptive FrankWofle for self - concordant
def FrankWolfewithSC(x, y, point = None, max_iter=200, tolerance=1e-100):
    M = 2  ##to be determined
    if point is None:
        # point = np.random.randn(x.shape[1], 1) * 1000
        point = np.array([[233., 1924., 182., -1191., -575., 643., 205., 385., 1028., 199.]]).T
    gaps = []
    loses = []
    for t in range(max_iter):
        gap, vk = FWSC_state(x, y,point)
        gaps.append(gap)
        loses.append(loss_function(x, y, point))
        if abs(gap) < tolerance:
            break
        Hssian = Hessian_LRgradient(x)
        e = M/2 * np.square(np.dot(np.dot(Hssian, point).T, point))
        tk = gap / (e * (gap + (4 * e)/(M ** M)))
        ak = min(1, tk)
        point = point - ak * vk
    return point, gaps, loses



def split_blocks(partition, shape):
    split_index = []
    while len(split_index) < partition - 1:
        random_index = random.randint(1, shape[0] - 1)
        if random_index not in split_index:
            split_index.append(random_index)
    split_index.append(shape[0])
    split_index.sort()
    return split_index


def FWSC_state(x,y, theta):
    gradient = gradient_entropyloss(theta, x, y)

    s = linear_oracle(gradient,theta)
    GAP = np.dot(gradient.T, theta - s)
    if GAP< 0:
        print("error GAP")
    return np.squeeze(GAP), theta - s


def linear_oracle(grad, theta):
    s_dot = np.dot(grad, theta.T)
    argmin = np.argmin(np.linalg.norm(s_dot, axis=1, keepdims=True))
    s = np.array(s_dot[argmin])
    return s.reshape(grad.shape)


def LR(x):
    return 1.0 / (1.0 + np.exp(-x))


def gradient_entropyloss(theta, x, y):
    size = x.shape[0]
    return (1 / size) * np.dot(x.T, LR(np.dot(x, theta)) - y)


def Hessian_LRgradient(x):
    size = x.shape[0]
    g2 = np.dot(x.T,x)
    return (1 / (size *size)) * g2


def loss_function(x, y, theta):
    y_1 = 1. / (1. + np.exp(-x.dot(theta)))
    return - np.sum(y * np.log(y_1) + (1 - y) * np.log(1 - y_1)) / len(y)


def get_blocks(x, partition = 3):
    blocks_i = split_blocks(partition, (x.shape[1], 1))
    previous_index = 0
    blocks_index = []
    for b_i in blocks_i:
        blocks_index.append((previous_index,b_i))
        previous_index = b_i
    return blocks_index


def display_performance(gaps, title, performance):
    plt.plot(gaps)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel(performance)
    plt.title(title)
    plt.grid()
    plt.show()

# This is wrong implementation, forget it.
# def FrankWolfe(learning_rate, x, y, max_iter=200, tolerance=1e-6):
#     #point = np.random.randn(x.shape[1], 1) * 0.001
#     point = np.zeros((x.shape[1], 1))
#     gaps = []
#     current_WX = np.dot(y, x)
#     for t in range(max_iter):
#         x_a = np.squeeze(np.dot(x, point))
#         gradient = np.dot(x.T, x_a) - current_WX
#         max_index = np.argmax(np.abs(gradient))
#         g_t1 = np.squeeze(np.dot(point.T, gradient))
#         g_t2 =gradient[max_index] * np.sign(-gradient[max_index]) * learning_rate
#         g_t = g_t1 - g_t2
#         gaps.append(g_t)
#         if g_t < tolerance:
#             break
#         Ldt2 = x[:, max_index] * np.sign(-gradient[max_index]) * learning_rate - x_a
#         stride = min(np.dot(Ldt2, y - x_a) / np.dot(Ldt2,Ldt2), 1.)
#         point = (1. - stride) * point
#         point[max_index] = point[max_index] + stride * np.sign(-gradient[max_index]) * learning_rate
#     return point, gaps