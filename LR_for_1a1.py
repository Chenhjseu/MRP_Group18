import numpy as np
from scipy.optimize import fmin_tnc
from sklearn.linear_model import LogisticRegression
import re
import FrankW
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

class LogisticR:
    def __init__(self, data_path):
        self.datapath = data_path
        self.parameters = {}
        self.dimension_S = 1
        self.dimension_E = 15

    def kernel(self, x, activation ="sigmoid"):
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))

    def loss_function(self, x, y, theta):
        y_1 = 1. / (1. + np.exp(-x.dot(theta)))
        return - np.sum(y * np.log(y_1) + (1 - y) * np.log(1 - y_1)) / len(y)
        # total_loss = -(1/size) * np.sum(np.log(self.kernel(np.dot(x, theta))) * y + (1-y) * np.log(1-self.kernel(np.dot(x, theta))))
        # return total_loss

    ### baseline for the compare the result
    def gradient(self, theta, x, y):
        size = x.shape[0]
        return (1 / size) * np.dot(x.T, self.kernel(np.dot(x, theta)) - y)

    def load_data(self):
        x = []
        y = []
        
        with open(self.datapath) as f:
            lines = f.readlines()
        for line in lines:
            #temp_line = re.split('\s\d+:', line)
            temp_line = list(filter(None, re.split('\s(\d+):\d+', line)))
            #temp_line = re.split(':+\d', line)
            #List_X = list(map(int, temp_line[self.dimension_S:self.dimension_E]))
            List_X = list(temp_line[self.dimension_S:self.dimension_E])
            if len(List_X) == 14:
                #y.append(int(temp_line[0]) - 1)
                #x.append(List_X)
                x.append(List_X)

                if int(temp_line[0]) == -1:
                    y.append(int(temp_line[0]) + 1)
                else:
                    y.append(int(temp_line[0]))

        return np.array(x), np.array(y).reshape(len(y), 1)

    #def process_gtd(self):
    #    x, y = self.load_data()
    #    self.parameters['theta'] = np.random.randn(x.shape[1], 1) * 0.1
    #    model, loss = self.LRGD(x, y, 1000)
    #    result = self.kernel(np.dot(x, model))
    #    y_predict = np.array(result >= 0.5, dtype='int')
    #    FrankW.display_performance(loss, "GD", "loss")
    #    return result

    def process_fw(self):
        x, y = self.load_data()
        point, gaps, losses = FrankW.FrankWolfe(14, x, y, max_iter= 1000)
        FrankW.display_performance(gaps, "FW", "FW_GAPS")
        FrankW.display_performance(gaps, "FW", "FW_Losses")

    def process_fwcs(self):
        x, y = self.load_data()
        point, gaps, losses = FrankW.FrankWolfewithSC(x, y)
        FrankW.display_performance(gaps, "FW_SC", "GAPS")
        FrankW.display_performance(losses, "FW_SC", "Losses")

    def process_BDfwcs(self):
        x, y = self.load_data()
        point, gaps, losses = FrankW.FW_BD(x, y)
        FrankW.display_performance(gaps, "FW_BD", "FW_GAPS")
        FrankW.display_performance(losses, "FW_BD", "FW_Losses")

    def LRGD(self, x, y, maxIteration, learning_rate = 0.1):
        n = np.shape(x)[0]
        theta = self.parameters['theta']
        iteration = 0
        y= y.reshape((n,1))
        performance = []
        while iteration <= maxIteration:
            theta = theta - learning_rate * self.gradient(theta, x, y)
            iteration += 1
            loss = self.loss_function(x, y, theta)
            performance.append(loss)
            if iteration % 100 == 0:
                print("iteration", str(iteration), "error rate:：", str(loss))
        return theta, performance

if __name__ == '__main__':
    # if you need change the file path, edit here
    LogR = LogisticR('a_a/a1a.txt')

    # you can remove the anotation to run

    #FW
    test = LogR.process_fwcs()

    #GD
    #test = LogR.process_gtd()

    #Block with FW
    test2 = LogR.process_BDfwcs()
