import pymc.gp as gp
from pymc.gp.cov_funs import matern
import numpy as np
from matplotlib.pyplot import normalize, get_cmap
from matplotlib.colors import hsv_to_rgb

class LearningModel(object):
    '''Gaussian process model for 2d input and 1d output'''

    def __init__(self, xmin,xmax,ymin,ymax):

        self.xbounds = np.array([xmin,xmax])
        self.ybounds = np.array([ymin,ymax])

        #make the mesh
        x_mesh = np.arange(xmin, xmax, 0.1)
        y_mesh = np.arange(ymin, ymax, 0.1)
        self.xx, self.yy = np.meshgrid(x_mesh, y_mesh)
        self.xy = np.concatenate((np.atleast_2d(self.xx.ravel()), np.atleast_2d(self.yy.ravel()))).T #this is the nx2 matrix with mesh points

        #set up gaussian process mean and covariance functions
        self.M = gp.Mean(self.meanfun)
        #might want to change these parameters
        self.C = gp.Covariance(eval_fun = matern.euclidean, diff_degree = 10.4, amp = 10.4, scale = 10.)

    def meanfun(self,xy):
        """
        mean function for the gp - here, it's just 0
        :param xy: nx2 dimensional array of inputs
        :output: nx1 dimensional array of zeros
        """
        return np.zeros((len(xy), 1))

    def sinxfun(self, xy):
        d = np.sqrt(xy[:, 0]**2+xy[:, 1]**2)
        return d*np.sin(d)

    def get_realization(self):
        r = gp.Realization(self.M, self.C)
        return np.reshape(r(self.xy), self.xx.shape)

    def get_realization_flat(self):
        r = gp.Realization(self.M, self.C)
        return r(self.xy)

    def get_mean(self):
        return self.M(self.xy)

    def collect_data(self):
        obs = np.random.uniform(-10,10,(10,2)) #10 points
        V = np.array([.002,.002]) #lower value makes the GP fit closer to the data, higher value makes it less important to do that.
        data = self.sinxfun(obs) #find the z-value of the points
        gp.observe(self.M, self.C, #current mean and covariance functions ->> it updates these!! (I think)
                obs_mesh=obs, #INPUT values for datapoints
                obs_V = V,  #NOISE?  Variance
                obs_vals = data) #actual OUTPUT values observed

learn = LearningModel(-10, 10, -10, 10)
#z = learn.get_realization_flat()

def scale_to_255(z):
    return (z-np.min(z))/(np.max(z)-np.min(z))

import pyprocessing as pyp
width = 200
height = 200


def setup():
    global width, height
    pyp.size(width, height)


def draw():
    global m, learn, z
    pyp.background(200,50)

    learn.collect_data()
    m = learn.get_mean()


    pyp.loadPixels()
    m = np.atleast_2d(m)
    norm = normalize(vmin=min(min(m)), vmax=max(max(m)))
    cmap = get_cmap('jet')
    m_normed = norm(m)
    rgba_data=cmap(m_normed)*255
    r = rgba_data[0,:,0].astype('uint32')
    g = rgba_data[0,:,1].astype('uint32')
    b = rgba_data[0,:,2].astype('uint32')
    a = rgba_data[0,:,3].astype('uint32')
    pyp.screen.pixels = a << 24 | r << 16 | g << 8 | b
    pyp.updatePixels()
    #imshow(np.reshape(z,learn.xx.shape))
    #show()
pyp.run()
