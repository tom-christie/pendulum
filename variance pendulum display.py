

#import pandas as pd
import numpy as np
#import sklearn as skl
#import scipy.stats as stats #STATS ARE IN THE SCIPY MODULE
#%pylab inline
#import pygame as pg
import math
import matplotlib as mpl
import numpy.random as random
import itertools
mpl.rcParams['image.interpolation'] = 'none'

# <codecell>

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

# <codecell>

#function to get 
def get_bin_lower_bounds(data,num_bins):
    '''calculates lower bound values for bins with approximately the same number of samples in each bin
        returns a numpy array with the values for bin lower-bound cutoffs
    '''
    num_samples = size(data)
    sorted_samples = np.sort(data)
    #just do linspace between the number of sample values with num_bins+1, and use all but the last value
    bin_indices = np.linspace(0,num_samples-1,num_bins+1)
    #bin_indices = np.int_(bin_indices) #convert to integers b/c we're going to use them as indices
    bin_lower_bounds = sorted_samples[bin_indices[:-1].astype(int)]
    #plot(sorted_samples)
    return bin_lower_bounds

def get_bin_bounds(data,num_bins):
    '''calculates lower bound values for bins with approximately the same number of samples in each bin
        returns a numpy array with the values for bin lower-bound cutoffs
    '''
    num_samples = size(data)
    sorted_samples = np.sort(data)
    #just do linspace between the number of sample values with num_bins+1, and use all but the last value
    bin_indices = np.linspace(0,num_samples-1,num_bins+1)
    #print bin_indices
    #bin_indices = np.int_(bin_indices) #convert to integers b/c we're going to use them as indices
    bin_bounds = sorted_samples[bin_indices.astype(int)]
    bin_bounds[-1] = bin_bounds[-1] + .00000001 #make it slightly higher so you can use less-than
    #plot(sorted_samples)
    
    return bin_bounds


## EMPOWERMENT CALCULATOR
import multiprocessing

##TIME STEP SIZE AND NOISE ARE IMPORTANT
results = []
def collect_results(result):
    global results
    results.extend(result)
    
def create_samples(state,actions,dt):
    plots = False
    import time
    starttime = time.time()
    #print 'getting samples...'
    #variables for computing empowerment
    
    N_BINS = 50; #number of bins for Monte-Carlo samples
    N_SAMPLES = 5000; #number of samples to add noise to to get bin bounds - split into N_BINS
    N_MC = 200

      
    state = np.array(state)
    state_size = len(state)
    samples_per_bin = N_MC/pow(N_BINS,state_size)
    N = actions.shape[0]
    lookahead = actions.shape[1]
    
    action_state_probabilities = np.zeros((N,N_BINS,N_BINS)) #transition probabilities #p((x,v)'_nu|a_nu) - somehow the dimensionality needs to be the same as the number of states. 1+2 for now
    action_state_probabilities_bin_bounds = np.zeros((N,state_size,N_BINS+1)) 
    action_state_counts = np.zeros((N_BINS,N_BINS))
    action_state_areas = np.zeros((N,N_BINS,N_BINS))
    #print "A...",time.time()-starttime
    #INITIALIZE
    #(a) p0(action_nu) = 1/N for nu=1...N.  That is, make the initial action policy uniform (we'll change this later)

    #print "B...",time.time()-starttime

    #for each action nu, make a transition matrix based on that action by taking samples and averaging them
    #it's a joint distribution for all values of the state given an action
    #the probabilities are all uninteresting 
    samples_for_action = {}
    for nu in xrange(N):
        samples = zeros((N_SAMPLES,state_size)) #get samples of both 
        test = PhysicsModel(state,dt)

        ###THIS IS WHERE THE DIFFERENCE IS FOR N-STEP
        samples = test.sample_steps_lookahead_fast(N_SAMPLES,actions[nu,:])  #######THIS IS WHAT TAKES FOREVER
        if plots:
            plt.figure(1)
            hist(samples[:,0])
            plt.figure(3)
            hist(samples[:,1])

        #store samples for each action in a dict for later
        samples_for_action[nu] = samples
        #get histogram bounds for all dimensions
    
        for dim in xrange(state_size):
            #print dim
            action_state_probabilities_bin_bounds[nu,dim,:] = get_bin_bounds(samples[:,dim],N_BINS) 
            if dim==0 and plots:
                plt.figure(2)
                plot(get_bin_bounds(samples[:,dim],N_BINS))
            
        d0 = diff(action_state_probabilities_bin_bounds[nu,0,:])
        d1 = diff(action_state_probabilities_bin_bounds[nu,1,:])
        d0 = np.reshape(d0,d0.shape + (1,))
        d1 = np.reshape(d1,d1.shape + (1,))
        action_state_areas[nu,:,:] = d0*d1.T
            
        temp = pow(action_state_areas[nu,:,:],-1) #assuming that the probabilities are inversely proportional to the areas... (divide by height too???????)######################
        action_state_probabilities[nu,:,:] = temp/sum(temp) #make them sum to 1
        #print 'sum',sum(action_state_probabilities[nu,:,:])

    
    
    #find prob for each of N_MC samples x'_nu for each of the actions a_mu
    MC_indices = random.randint(0,N_SAMPLES,N_MC)
    all_sample_probs = np.zeros((N,N,N_MC)) #these are the probabilities of p(x'_nu | a_mu) for each combo of nu/mu and for all the samples
    
    
    ###THIS IS WHATS TAKING A LONG TIME
    #for each action
    count = 0    
    for nu in xrange(N):
        samp = samples_for_action[nu] 
        for index1 in xrange(len(MC_indices)): #for each sample under action nu
            s = samp[MC_indices[index1],:]
            for mu in xrange(N): #find the probability that each of these happened under action MU
                prob=bins_to_probs(action_state_probabilities_bin_bounds,action_state_probabilities,mu,s) #find the probability that x'_nu would have occurred under action mu
                all_sample_probs[nu,mu,index1]=prob
                count = count + 1

    
    if 0:
        plt.figure(4)
        plt.imshow(all_sample_probs[:,:,1])
        plt.colorbar()
    return all_sample_probs
    
    

def BlahutArimotoContinuous(all_sample_probs):
    '''Calculate an empowerment estimate for each action.
    INPUT
    all_sample_probs: all_sample_probes(a1,a2,x) = p(x_a1|a2), for a sample x.  
    just interesting for doing the MC thing.
    So the number of x's corresponds to the number of samples. it's an (actions x actions x samples) matrix. 

    Basically the only difference is that you ask the simulator to look several steps ahead and return the result, so you get
    p(x'_t+n|a_n) rather than p(x'_t+1|a_1)
'''
    
    TOL = pow(10,-5) #how close to get in the estimate
    MAX_ITERATIONS = 150
    N = all_sample_probs.shape[0]
    N_MC = all_sample_probs.shape[2]
    
    p0 = np.zeros((N))
    for nu in xrange(N):
        p0[nu] = 1.0/N
    
    pk = p0
    #ITERATE
    k=0
    c = np.zeros(MAX_ITERATIONS)
    z = np.zeros(MAX_ITERATIONS)
    c[0] = -1 #just to set it to something very small
    while (k<4 or abs(c[k-2]-c[k-3])>=TOL) and k < MAX_ITERATIONS:
        #(a)
        z[k] = 0
        c[k-1]=0 #previous is set to ck1
        
        #(b)
        for nu in xrange(N): #stupid zero-indexing
            #now use the probabilities of the samples (or just a subset of them)
            temp = 0
            for index in xrange(N_MC):
                temp = temp+log2(all_sample_probs[nu,nu,index]/np.dot(all_sample_probs[nu,:,index],pk))
            d = temp/N_MC 
            #print d
            
            c[k-1] = c[k-1]+pk[nu]*d #current is set to ck
            pk[nu] = pk[nu]*exp(d)
            z[k] = z[k]+pk[nu]
        #(c)
        for nu in xrange(N):
            pk[nu] = pk[nu]/z[k]
        k = k+1
        #print 'k',k
        #print 'ck',c[k-2]
        #print 'ck1',c[k-3]
        #print c
        #print 'action distribution = ',pk
    print 'qw Empowerment = ',c[k-2]
    #print "threshold",abs(c[k-2]-c[k-3])
    return c[k-2]#return(pk,ck1)
    
#MAYBE THERE'S A WAY TO DO THIS ONCE AND USE THE RESULTS MORE QUICKLY...    
def bins_to_probs_mp(bins,action_state_probabilities,action_index,value,que): 
    '''takes bin boundaries and associated probabilities, and looks up a prob corresponding to the value passed
    basically finds p(x'_nu | a_mu).  
    input - 
    bins: Location of bin bounds for each action/dimension combination #actions x state dimensionality x number of bins+1.  
    action_state_probabilities: joint probabilities of each bin range (x,v) for each action
    action_index: which action MU we're looking at the probability of x'_nu happening
    value: state achieved FROM ACTION NU

    '''
    #find the probability that x'_nu would have occurred under action mu
    loc = np.zeros(value.shape) #bin indices
    for dim in xrange(len(value)): #for each dimension in the state
        temp = nonzero(bins[action_index,dim,:]<=value[dim])#determine the bin in mu the state would be in - returns a tuple
        if len(temp[0]) == 0 or len(temp[0])==len(bins[action_index,dim,:]): #if outside the bin range, the probability is zero (or basically zero)
            return 0

        loc[dim] = temp[0][-1] #otherwise, this gives the bin number from mu for the specific dimension
    que.put(action_state_probabilities[tuple(concatenate([np.array([action_index]),loc.astype(int)]))])#the joint probability for the entire state  associated with the the bin in mu
    
    
#MAYBE THERE'S A WAY TO DO THIS ONCE AND USE THE RESULTS MORE QUICKLY...    
def bins_to_probs(bins,action_state_probabilities,action_index,value): 
    '''takes bin boundaries and associated probabilities, and looks up a prob corresponding to the value passed
    basically finds p(x'_nu | a_mu).  
    input - 
    bins: Location of bin bounds for each action/dimension combination #actions x state dimensionality x number of bins+1.  
    action_state_probabilities: joint probabilities of each bin range (x,v) for each action
    action_index: which action MU we're looking at the probability of x'_nu happening
    value: state achieved FROM ACTION NU

    '''
    #find the probability that x'_nu would have occurred under action mu
    loc = np.zeros(value.shape) #bin indices
    for dim in xrange(len(value)): #for each dimension in the state
        temp = nonzero(bins[action_index,dim,:]<=value[dim])#determine the bin in mu the state would be in - returns a tuple
        if len(temp[0]) == 0 or len(temp[0])==len(bins[action_index,dim,:]): #if outside the bin range, the probability is zero (or basically zero)
            return 0

        loc[dim] = temp[0][-1] #otherwise, this gives the bin number from mu for the specific dimension
    return action_state_probabilities[tuple(concatenate([np.array([action_index]),loc.astype(int)]))]#the joint probability for the entire state  associated with the the bin in mu




class Controller():
    '''controls the pendulum object, 
        decides what to do next based on info available.'''
    def __init__(self):
        self.actions = np.array([-5, -2.5, 0, 2.5, 5])
        #self.actions = np.array([-5,0,5])
        #self.actions = np.linspace(-1,1,10)        
        
        #self.actions = np.array([-0.5, -0.25, 0, 0.25, 0.5])
        self.actions = self.actions/5
        self.lookahead = 3
        self.state = np.array([np.pi,0]) #for the pendulum, it's position/angular velocity. I think pi is at the bottom. 
        self.dt = 1.0/5
        self.action_taken = None
        
    def step(self):
        '''main function. updates the state.  does the following things
        1) simualtes each actoon to get x'
        2) for each x', calls the BA algorithm to get n-step empowerment (for each combo of successor actions)
        3) executes the action with the highest empowerment
        4) determines the resulting state
        '''
        
        #SIMULATE EACH ACTION TO GET X'
        x_hypothetical_states = np.zeros((len(self.actions),len(self.state)))
        for action in xrange(len(self.actions)):
            test = PhysicsModel(self.state,self.dt)
            test.step(self.actions[action])   
            #print test.state
            x_hypothetical_states[action,:] = test.state
        #print x_hypothetical_states
        #NOTE: DON'T REALLY UNDERSTAND THE RESULTS, BUT THE SIMULATOR SEEMS TO WORK SO JUST GO WITH IT...FOR NOW
        
        #FOR EACH X', CALL BA ALGORITHM TO GET N-STEP EMPOWERMENT (FOR EACH COMBINATION OF ACTIONS
        #make a combo of actions
        action_combos = np.zeros( (pow(len(self.actions),self.lookahead), self.lookahead))  # action series X lookahead 
        counter = 0
        for action_combo in itertools.product(self.actions,repeat=self.lookahead):
            for index in range(len(action_combo)):
                #print index
                action_combos[counter,index] = action_combo[index]
            counter = counter + 1
        #print action_combos
        
        self.variance_for_successor_states = np.zeros(len(self.actions))
        #print len(x_hypothetical_states)
        #print x_hypothetical_states.shape
        index = -1
        for hypothetical_state in xrange(len(x_hypothetical_states)):
            #print "qw hypothetical state",x_hypothetical_states[hypothetical_state,:]
            index = index + 1
            

            #for each action
            predicted_states = np.zeros((pow(len(self.actions),self.lookahead),2))
            counter = -1
            for action in action_combos:
                counter = counter + 1
                #predict the next state
                test = PhysicsModel(x_hypothetical_states[hypothetical_state,:],self.dt)
                predicted_states[counter,:] = test.sample_steps_lookahead_fast(1,action)
                
                #calculate the variance of the predictions
            
            #print predicted_states
            v=np.var(predicted_states,axis=0)
            #print v
            self.variance_for_successor_states[index] = v[0]+v[1]
        #print self.variance_for_successor_states
        np.set_printoptions(threshold=np.nan)
        self.action_taken = np.argmax(self.variance_for_successor_states);
        self.state = x_hypothetical_states[np.argmax(self.variance_for_successor_states),:]
            
            
            
            
            
############## PYPROCESSING ################

import pyprocessing as pyp
width = 600;
height = 700;
xcenter = width/2;
ycenter = height/2;
pendulum_length = 200
indicators = []
num_actions = 5
indicator_width = 40
step_counter = 0

def setup():
    pyp.size(width,height)
    pyp.ellipseMode(pyp.CENTER)
    pyp.rectMode(pyp.CENTER);
    indicator_spacing = np.linspace(xcenter-indicator_width*num_actions/2, xcenter+indicator_width*num_actions/2,num_actions)
    print indicator_spacing
    for i in range(num_actions):
        indicators.append(Indicator(indicator_spacing[i],height-50))
    print indicators
    step_counter = 0

class Indicator(object):
    def __init__(self,xposition,yposition):
        self.size = 20
        self.xposition = xposition
        self.yposition = yposition
        self.fillcolor = pyp.color(200,0,0)
        self.emptycolor = pyp.color(150,100)

    def fill(self):
        pyp.fill(self.fillcolor);
        pyp.rect(self.xposition, self.yposition,self.size,self.size);

    def empty(self):
        pyp.fill(self.emptycolor)
        pyp.rect(self.xposition, self.yposition,self.size,self.size);
    
        

res = []
z = Controller()
variance_history = np.zeros((100,5)) #has to be the same width as the 
z.state = np.array([np.pi+.2,0])
velocity = None
angle = 0

def draw():
    z.step()
    angle = (2*np.pi-z.state[0]-np.pi/2)
    pyp.background(200,50)
    pyp.line(xcenter,ycenter,xcenter+ pendulum_length*np.cos(angle),ycenter+ pendulum_length*np.sin(angle));
    pyp.fill(255)
    pyp.ellipse(xcenter+ pendulum_length*np.cos(angle),ycenter+ pendulum_length*np.sin(angle),100,100)
    action_taken = z.action_taken
    for i in range(len(indicators)):
        #print action_taken
        if i == action_taken:
            indicators[i].fill()
        else:
            indicators[i].empty()
    # pyp.stroke(126);
    # pyp.line(85, 20, 85, 75);
    # pyp.stroke(255);
    # pyp.line(85, 75, 30, 75);
#    for i in range(100):
#        #print i
#        #print "STATE",z.state 
#        res.append(z.state[0])
#        variance_history[i,:] = z.variance_for_successor_states 


pyp.run()



#PLOTS TRAJECTORY UNDER VARIANCE-BASED CONTROL
if 0:
    res = []
    z = Controller()
    variance_history = np.zeros((100,5)) #has to be the same width as the 
    z.state = np.array([np.pi+.2,0])
    z.step()
    for i in range(100):
        print i
        print "STATE",z.state
        z.step()
        res.append(z.state[0])
        variance_history[i,:] = z.variance_for_successor_states 
    #plot(res)
#    print z.variance_history
    #imshow(variance_history)
    
    
#DISPLAYS EMPOERMENT/VARIANCE OF POINTS IN STATE-SPACE
if 0:
    import itertools

    #actions
    actions = np.array([-5, -2.5, 0, 2.5, 5])
    lookahead = 3
    dt = 1.0/5
    #make list of action combinations
    action_combos = np.zeros( (pow(len(actions),lookahead), lookahead))  # action series X lookahead 
    counter = 0
    for action_combo in itertools.product(actions,repeat=lookahead):
        for index in range(len(action_combo)):
            #print index
            action_combos[counter,index] = action_combo[index]
        counter = counter + 1
    
    
    #states
    x = np.linspace(0,2*np.pi,20)
    v = np.linspace(-np.pi,np.pi,20)
    #for each state
    figure()
    results = zeros((400,3))
    rescounter = -1
    for state in itertools.product(x,v):
        rescounter = rescounter + 1
        #print 'state',state
        #make an array to hold the predicted states for the actions
        predicted_states = np.zeros((pow(len(actions),lookahead),2))
        #for each action
        counter = -1
        for action in action_combos:
            counter = counter + 1
            #predict the next state
            test = PhysicsModel(np.array(state),dt)
            predicted_states[counter,:] = test.sample_steps_lookahead_no_noise(1,action)
        #calculate the variance of the predictions
        v=np.var(predicted_states,axis=0) #????
        #print 'variances',v
    #    print results[rescounter]
        results[rescounter,:2] = state
        #print state
        results[rescounter,2] = v[0]
        #find the empowerment for the state also
        #print action_combos
        #z = create_samples(array(state), action_combos, dt)
        #print z.shape
        #emp = BlahutArimotoContinuous(z)
        #print 'empowerment',emp
        scatter(results[:,0],results[:,1],s=60,c=results[:,2],marker='s',edgecolors=None)
        #results[rescounter,2] = emp
        #plot(v[0]+v[1],emp)

