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
        self.dimension_E = 11

    def kernel(self, x, activation = "sig   moid"):
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

    def fit(self, x, y, theta):
        opt_weights =fmin_tnc(func=self.loss_function, x0=theta, fprime=self.gradient, args=(x, y.flatten()))
        return opt_weights[0]

    def load_data(self):
        x = []
        y = []
        with open(self.datapath) as f:
            lines = f.readlines()
        for line in lines:
            temp_line = re.split('\s\d+:', line)
            List_X = list(map(float, temp_line[self.dimension_S:self.dimension_E]))
            if len(List_X) == 10:
                y.append(int(temp_line[0]) - 1)
                x.append(List_X)
        return np.array(x), np.array(y)

    def process_gtd(self):
        x, y = self.load_data()
        self.parameters['theta'] = np.random.randn(x.shape[1], 1) * 0.1
        model, loss = self.LRGD(x, y, 1000)
        result = self.kernel(np.dot(x, model))
        y_predict = np.array(result >= 0.5, dtype='int')
        FrankW.display_performance(loss, "GD")
        return result

    def process_fw(self):
        x, y = self.load_data()
        point, gaps = FrankW.FrankWolfe(10, x, y, max_iter= 1000)
        FrankW.display_performance(gaps, "FW_gaps")

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

    def test(self):
        x, y = self.load_data()
        L1_LR = LogisticRegression(C=0.1, penalty='l2', tol=0.01)
        L1_LR.fit(x, y)
        L1_LR_predict = L1_LR.predict(x)
        print(L1_LR_predict)

    # todo: split the blocks by kmeans
    # def split_blocks_D(self, start_position = 2):
    #     x, y = self.load_data()
    #     model = KMeans()
    #     visualizer = KElbowVisualizer(
    #         model, k=(start_position, x.shape[1]), metric='calinski_harabasz', timings=False, locate_elbow=False
    #     )
    #     visualizer.fit(x.T)
    #     split_blocks = visualizer.k_scores_.index(max(visualizer.k_scores_)) + start_position
    #     kmeans = KMeans(n_clusters=split_blocks, random_state=0).fit(x.T)
    #     for i in range(0,max(kmeans.labels_) + 1):
    #         block_index =



if __name__ == '__main__':
    # if you need change the file path, edit here
    LogR = LogisticR('covtype_bin')

    # you can remove the anotation to run

    #FW
    test = LogR.process_fw()

    #GD
    #test = LogR.process_gtd()
    print(test)
