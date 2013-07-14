import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(1)


def f(points):
    """The function to predict.
    :param points:
    """
    distances = np.zeros((points.shape[0],1))
    for i in range(len(points)):
        #print points[i,:], points[i,:]**2
        distances[i] = np.sqrt(np.sum(points[i,:]**2))
    return distances * np.sin(distances)

#----------------------------------------------------------------------

# MESH
# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x_mesh = np.arange(-10, 10, 0.1)
y_mesh = np.arange(-10, 10, 0.1)
xx, yy = np.meshgrid(x_mesh, y_mesh)
mesh_points = np.vstack([xx.ravel(), yy.ravel()]).T  # #CONVERTS MESHGRID INTO LIST OF POINTS
z_mesh = f(mesh_points)
#
# fig = pl.figure(1)
# ax = fig.gca(projection='3d')
z_mesh = np.sqrt(xx**2+yy**2) * np.sin(np.sqrt(xx**2+yy**2))
# #pl.contourf(x_mesh,y_mesh,np.sqrt(xx**2+yy**2) * np.sin(np.sqrt(xx**2+yy**2)),100,alpha=.8)
# surf = ax.plot_surface(xx,yy, z_mesh,rstride=5, cstride=5, cmap=cm.coolwarm,
#         linewidth=1, antialiased=False, shade = True)
# ax.set_zlim(-12, 12)
# fig.colorbar(surf, shrink=0.5, aspect=5)
#

# TODO Use alpha value to show width of confidence interval???



#  First the noiseless case
all_data = 20*(np.random.rand(100,2)-.5)  #array of points in (-10,10)
fig2 = pl.figure(2)
for i in range(1,100):
    fig2.clf()
    data = all_data[:i,:]
    #print data
# #X = np.random.rand(100,2)
    target = f(data)
    #
    # Instanciate a Gaussian Process model
    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,random_start=100) #kernel is "corr"
    #gp = GaussianProcess(corr='squared_exponential',theta0 = .9)
    #gp = GaussianProcess(corr='squared_exponential')
    # Fit to data using Maximum Likelihood Estimation of the parameters
    print data, target
    gp.fit(data, target) # X is input and y is output


    # # # Make the prediction on the meshed x-axis (ask for MSE as well)
    # z_pred = gp.predict(mesh_points)
    # z_pred = np.reshape(z_pred,z_mesh.shape)
    # #print xx.shape
    # #print yy.shape
    # #print z_pred.shape
    #
    #
    # ax2 = fig2.gca(projection='3d')
    # surf2 = ax2.plot_surface(xx,yy, z_pred,rstride=1, cstride=1, cmap=cm.coolwarm,
    #         linewidth=0, antialiased=False, shade = True,alpha=0.1)
    # ax2.set_zlim(-12, 12)
    # ax2.scatter(data[:,0],data[:,1],target)
    # fig2.savefig('omnicient' + str(i) + '.png')
#pl.show()
# sigma = np.sqrt(MSE)
#
# # Plot the function, the prediction and the 95% confidence interval based on
# # the MSE









# fig = pl.figure()
# pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
# pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
# pl.plot(x, y_pred, 'b-', label=u'Prediction')
# pl.fill(np.concatenate([x, x[::-1]]),
#         np.concatenate([y_pred - 1.9600 * sigma,
#                        (y_pred + 1.9600 * sigma)[::-1]]),
#         alpha=.5, fc='b', ec='None', label='95% confidence interval')
# pl.xlabel('$x$')
# pl.ylabel('$f(x)$')
# pl.ylim(-10, 20)
# pl.legend(loc='upper left')
# pl.show()
#
