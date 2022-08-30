#!/usr/bin/env python
'''
Plot contact probability from a pickle output file by AF2Complex
'''

import numpy as np
import os,sys
import matplotlib.pyplot as plt
import scipy.special
import pickle


####
if len(sys.argv) < 2:
    print("python", sys.argv[0], "model_pickle_file")
    exit(1)

pickle_file = sys.argv[1]
pickle_file = os.path.expanduser( pickle_file )

if not os.path.exists(pickle_file):
    print(f"Error: {pickle_file} does not exist")
    sys.exit(1)

d = pickle.load(open(pickle_file,'rb'))
probs = scipy.special.softmax(d['distogram']['logits'], axis=-1)
cnt_prob = np.sum(probs[...,:19], axis=-1)

complex_size = cnt_prob.shape[0]
chain_breaks = []
for i in range(complex_size-1):
    if cnt_prob[i,i+1] < 0.9:
        chain_breaks.append(i)

print(f"Found chain break position: {np.array(chain_breaks)+1}")

fig = plt.figure()
ax = plt.gca()

# draw horizontal and verticl lines at chain break points
for term in chain_breaks:
    ax.hlines(y=term, xmin=0, xmax=complex_size, linewidth=0.5, color='white')
    ax.vlines(x=term, ymin=0, ymax=complex_size, linewidth=0.5, color='white')

#caxes = ax.matshow(cnt_prob, cmap=plt.get_cmap('cividis'))
caxes = ax.matshow(cnt_prob, cmap=plt.get_cmap('viridis'))
fig.colorbar(caxes)



plt.show()
