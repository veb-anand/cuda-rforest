import pandas as pd
import numpy as np
import sys

if (len(sys.argv) != 3):
    print('Note: you can pass in num_points and num_features from the shell ' + 
        'like:\n\tpython make_data.py 2000 5\n')
    num_points = int(input('Number of points: '))
    num_features = int(input('Number of features: '))
else:
    num_points = int(sys.argv[1])
    num_features = int(sys.argv[2])

data = np.random.rand(num_points, num_features)
data[:, 0] = np.random.randint(2, size=data.shape[0])
pd.DataFrame(data).to_csv('../cuda/data/data.csv', index=False, 
                          line_terminator='\n', header=False)
