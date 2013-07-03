__author__ = 'Tom'

# the purpose of this class is to keep track of observations/transitions from the point of view of the pendulum and
# use a gaussian process to model the transitions.
# for each dimension in the state vector x = [x1 x2], a gaussian process regression is done using all other dimensions.
# for example, x1' = f1(x1,x2) and x2' = f2(x1,x2) both need to be computed.
# In general, for an N-dimensional state vector x, we'll need to fit N gaussian processes