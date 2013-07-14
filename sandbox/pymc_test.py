__author__ = 'Tom'

#FUNCTION TO DRAW FROM - this is what we're trying to model

from matplotlib.pyplot import clf, subplot, figure, cm, title, show
from numpy import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot

def sinxfun(x=0,y=0):
    ''' for each (x,y) point, it returns d*sin(d), where d is the distance from the origin.'''
    d = sqrt(x**2+y**2)
    return d*sin(d)

def meanfun(xy):
    return zeros((len(xy),1))


def rungp():
    #DEFINE GAUSSIAN PROCESS
    #mean is just 0
    MM = Mean(meanfun)
    #covariance is what was used in the example above
    CC = Covariance(eval_fun = matern.euclidean, diff_degree = 10.4, amp = 10.4, scale = 10.) #big diff_degree

    ##MAKING THE MESH
    x_mesh = np.arange(-10, 10, 0.1)
    y_mesh = np.arange(-10, 10, 0.1)
    xx, yy = np.meshgrid(x_mesh, y_mesh)

    z_mesh = sinxfun(xx,yy)

    ##PLOTTING

    fig = figure(figsize=(16,30))
    clf()

    subplot(5,1,1)
    ax = fig.add_subplot(511,projection='3d')
    ax.plot_surface(xx, yy, z_mesh, cmap=cm.jet,vmin=-15,vmax=15)
    ax.view_init(elev=50, azim = 120) #how high up you see it from, and what angle you see it from
    title('what we\'re trying to estimate')


    xy = concatenate((atleast_2d(xx.ravel()),atleast_2d(yy.ravel()))).T

    #REALIZATION WITH NO DATA
    #r = Realization(MM,CC)
    #subplot(5,1,2)
    #ax = fig.add_subplot(512,projection='3d')
    #ax.plot_surface(xx,yy, reshape(r(xy),xx.shape),alpha=.3,cmap=cm.jet,vmin=-15,vmax=15)
    #ax.set_zlim(-15,15)
    #title('realization1')

    #r = Realization(MM,CC)
    #subplot(5,1,3)
    #ax = fig.add_subplot(513,projection='3d')
    #ax.plot_surface(xx,yy, reshape(r(xy),xx.shape),alpha=.3,cmap=cm.jet,vmin=-15,vmax=15)
    #ax.set_zlim(-15,15)
    #title('realization 2')

    ### INCLUDE SOME ACTUAL DATA POINTS
    obs = np.random.uniform(-10,10,(5,2)) #5 points
    V = array([.002,.002]) #lower value makes the GP fit closer to the data, higher value makes it less important to do that.
    data = sinxfun(obs[:,0],obs[:,1]) #find the x-value of the points
    observe(MM, CC, #current mean and covariance functions ->> it updates these!! (I think)
            obs_mesh=obs, #INPUT values for datapoints
            obs_V = V,  #NOISE?  Variance
            obs_vals = data) #actual OUTPUT values observed
    #r = Realization(MM,CC)

    subplot(5,1,4)
    ax = fig.add_subplot(514,projection='3d')
    ax.plot_surface(xx,yy, reshape(MM(xy),xx.shape),alpha=.3,cmap=cm.jet,vmin=-15,vmax=15)
    ax.set_zlim(-15,15)
    ax.plot(obs[:,0],obs[:,1],data,'.k',markersize=20)
    title('after data')



    obs = np.random.uniform(-10,10,(500,2)) #500 points
    V = array([.0002,.0002])
    data = sinxfun(obs[:,0],obs[:,1]) #find the x-value of the points
    observe(MM, CC, #current mean and covariance functions ->> it updates these!! (I think)
            obs_mesh=obs, #INPUT values for datapoints
            obs_V = V,  #NOISE?  Variance
            obs_vals = data) #actual OUTPUT values observed
    #r = Realization(MM,CC)

    subplot(5,1,5)
    ax = fig.add_subplot(515,projection='3d')
    ax.plot_surface(xx,yy, reshape(MM(xy),xx.shape),alpha=.8,cmap=cm.jet,vmin=-15,vmax=15)
    ax.set_zlim(-15,15)
    ax.plot(obs[:,0],obs[:,1],data,'.k',markersize=5)
    title('after data')

    #show()

    #ax.plot_surface(xx, yy, z_mesh, cmap=cm.jet)
    # #how high up you see it from, and what angle you see it from

import cProfile
cProfile.run('rungp()')
