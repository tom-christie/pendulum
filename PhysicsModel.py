
import numpy as np
import math
import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'none'



class PhysicsModel():
    '''
    This model encodes the transition matrix p(x'|x,a).  The action is given as input to "step", and a record of the current state is maintained.
    Generic model of system dynamics.
    Specific equations should be limited to the predict function.
    Some things may need to be hand-coded until I can come up with a better data structure.
    '''

    #keep track of -
    # - state variables (pass in dimensions), plus any limitations to them
    # - control variable applied
    # - update delta_t (pass in delta_t)

    def __init__(self,initial_state,dt):
        #instantaneous
        self.state = np.array(initial_state)  #state vector
        self.n = len(initial_state)
        self.dt = dt #time step in seconds
        self.t = 0 #global time
        self.counter = 0
        self.sample_noise = 0.00001
        self.system_noise = 0.0000001

        #history
        self.x_history = np.zeros((10000,1))
        self.v_history = np.zeros((10000,1))

    def disp(self):
        print self.state
        print self.dt #time step in seconds
        print self.t #global time

    def step(self,u):
        #get mean transition estimate
        self.state = self.step_rk4(self.state[0],self.state[1],self.dt,u)
        #add noise
        #self.state = self.state + np.random.normal(0, self.system_noise, self.n)
        self.t  = self.t + self.dt

        try:
            self.x_history[self.counter] = self.state[0]
            self.v_history[self.counter] = self.state[1]
        except Exception:
            print Exception

        self.counter = self.counter + 1



    def sample_steps(self,num_samples,action):
        '''num_samples are just the number of samples to get
        action is the action to take, i.e. the actuation to send to step_rk4
        '''
        samples = np.zeros((num_samples,self.n))
        for i in xrange(num_samples):
            #noise = np.random.normal(0, self.sample_noise, self.n)
            #noise[-1]=noise[-1]/10
            samples[i,:] = self.step_rk4(self.state[0],self.state[1],self.dt,action)
        return samples

    def sample_steps_lookahead(self,num_samples,actions,steps_to_hold_action=1):
        '''num_samples: the number of samples to create
           actions: the action sequence to simulate.
               need to do this iteratively, so that the
               output from an action is the input for the next.
            steps_to_hold_action: number of steps
        '''

        ###FORGET ABOUT STPS_TO_HOLD for now

        samples = np.zeros((num_samples,self.n))
        for i in xrange(num_samples):
            #get one sample
            state_to_track = self.state
            #iterate over all actions into the future
            for action in actions:
                noise = np.random.normal(0, self.sample_noise, self.n)
                state_to_track = state_to_track + noise
                state_to_track = self.step_rk4(state_to_track[0],state_to_track[1],self.dt,action)

            samples[i,:] = state_to_track
        return samples

    def sample_steps_lookahead_fast(self,num_samples,actions,steps_to_hold_action=1):
        '''num_samples: the number of samples to create
           actions: the action sequence to simulate.  <--ONLY ONE ACTION SEQUENCE (this function is called separately for each)
               need to do this iteratively, so that the
               output from an action is the input for the next.
            steps_to_hold_action: number of steps
        '''

        prediction = np.zeros(actions.shape)  #dimensionality of the state
        #generate the predictions based on the actions
        state_to_track = self.state
        for action in actions:
            state_to_track = self.step_rk4(state_to_track[0],state_to_track[1],self.dt,action)
        #print "qw state",state_to_track
        #duplicate the state_to_track
        samples = np.reshape(np.repeat(state_to_track,num_samples),(len(state_to_track),num_samples)).T
        #print samples

        #add noise
        samples = samples + np.random.normal(0, self.sample_noise, samples.shape)
        return samples

    def sample_steps_lookahead_no_noise(self,num_samples,actions,steps_to_hold_action=1):
        '''num_samples: the number of samples to create
           actions: the action sequence to simulate.  <--ONLY ONE ACTION SEQUENCE (this function is called separately for each)
               need to do this iteratively, so that the
               output from an action is the input for the next.
            steps_to_hold_action: number of steps
        '''

        prediction = np.zeros(actions.shape)  #dimensionality of the state
        #generate the predictions based on the actions
        state_to_track = self.state
        for action in actions:
            state_to_track = self.step_rk4(state_to_track[0],state_to_track[1],self.dt,action)
        #print "qw state",state_to_track
        #duplicate the state_to_track
        samples = np.reshape(np.repeat(state_to_track,num_samples),(len(state_to_track),num_samples)).T
        #print samples

        #add noise
        #samples = samples + np.random.normal(0, self.sample_noise, samples.shape)
        return samples



    def step_rk4(self,x, v, dt, u):
        """Returns (position, velocity) tuple after
        time dt has passed, given control variable u.

        INPUTS
        x: initial position (number-like object)
        v: initial velocity (number-like object)
        a: acceleration function a(x,v,dt,u) (must be callable)
        dt: timestep (number)
        u: is control variable

        OUTPUTS
        xf: final x-value after dt has passed.
        vf: final y-value after dt has passed.
        """

        x1 = x
        v1 = v
        a1 = self.accel(x1, v1, u)

        x2 = x + 0.5*v1*dt
        v2 = v + 0.5*a1*dt
        a2 = self.accel(x2, v2, u)

        x3 = x + 0.5*v2*dt
        v3 = v + 0.5*a2*dt
        a3 = self.accel(x3, v3, u)

        x4 = x + v3*dt
        v4 = v + a3*dt
        a4 = self.accel(x4, v4, u)

        xf = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
        vf = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

        return np.array([xf, vf])

    #this is the system-dependent function to evaluate the acceleration after
    def accel(self,x,v,u):
        '''FOR A PENDULUM, calculates (angular) acceleration based on current angular position & velocity.

        INPUTS
        x: initial position (number-like object)
        v: initial velocity (number-like object)
        u: is control variable, i.e. force applied to the pendulum

        OUTPUTS
        af: final acceleration
         '''
        g = 9.8 #gravity
        m = 1 #mass
        li = 1 #length of pendulum i
        mu = 0.05 #coefficient of friction
        af = 1/(m*li*li) * (-1*mu*v + m*g*li*math.sin(x) + u)
        return af

if 0:
    test = PhysicsModel([np.pi,0],1.0/5)
    for i in xrange(50):
        test.step(0)
    plot(test.x_history[1:test.counter])
    plot(test.v_history[1:test.counter])
    print test.x_history[test.counter-1]

