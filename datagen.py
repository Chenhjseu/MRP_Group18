import random
import numpy as np
from datetime import datetime

### Generating the features

### m is the number of datapoints and N is the number of features
def data_generator(m, N):

    combined = []

    x_array = np.random.uniform(0, 1, (m, N))
    x = x_array / np.linalg.norm(x_array)

    ### Generating the classes
    y = []
    values = [1, 0]

    for i in range(m):
        y.append(random.choice(values))

    combined = np.column_stack((x, y))

   # with open('datagen.txt', 'w') as file:
   #     for item in combined:
   #         file.write('%s\n' % item)

    # with open('datagen.txt', 'w') as file:
    #     for item in combined:
    #         s = " ".join(map(str, item))
    #         file.write(s + '\n')

    return np.array(x), np.array(y)

### To validate that the eucledian norm equals to 1
#euclideanNorm = x.T.dot(x)
#euclideanNorm

# start_time = datetime.now()
#
# ### Specify number of datapoints and features
# data_generator(1000, 3000)
#
# end_time = datetime.now()
#
# print('Duration: {}'.format(end_time - start_time))
