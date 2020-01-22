# (c) 2020 alherit
# This code is licensed under MIT license (see LICENSE.txt for details)

import numpy as np
from pymlba import predict


D = np.mgrid[1:9.1:.1, 1:9.1:.1].reshape(2, -1).T

X = np.tile([4.,6.],[D.shape[0],1])
Y = np.tile([6.,4.],[D.shape[0],1])




M = np.concatenate([X,Y,D], axis=1).reshape(-1,3,2)

probs = predict(M, 5, .2, .4, 5.)

probs = np.stack(probs, axis=0)

prefX = probs[:,0] /(probs[:,0] + probs[:,1])


from pylab import imsave

Z = prefX.reshape(int(round(np.sqrt(len(prefX)))),-1).T

im = imsave("true.pdf",Z,cmap='gray', vmin=0., vmax=1., origin="lower")


from pylab import imshow, text, plot, savefig, scatter
import matplotlib
matplotlib.use('Agg')

plot([1, 9], [9, 1], '--', color="w", lw=1, zorder=5)
scatter([4,6,3,2,3.5], [6,4,5,8,6.5], marker =".", color="k", zorder=10)

fs= 20
text(4,6,"$a$",fontsize=fs)
text(6,4,"$b$",fontsize=fs)
text(3,5,"$c_A$",fontsize=fs)
text(2,8,"$c_C$",fontsize=fs)
text(3.5,6.5,"$c_S$",fontsize=fs)


im = imshow(Z, cmap='gray', vmin=0., vmax=1., origin="lower", extent=[1,9,1,9])
savefig("true_decorated.pdf",bbox_inches='tight')

print(probs)



