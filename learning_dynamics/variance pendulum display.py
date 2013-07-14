import itertools

import numpy as np
import matplotlib as mpl
import numpy.random as random

from PhysicsModel import PhysicsModel

from Controller import Controller

mpl.rcParams['image.interpolation'] = 'none'


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




import pyprocessing as pyp
width = 600
height = 700
xcenter = width/2
ycenter = height/2
pendulum_length = 200
indicators = []
num_actions = 5
indicator_width = 40
step_counter = 0

def setup():
    pyp.size(width, height)
    pyp.ellipseMode(pyp.CENTER)
    pyp.rectMode(pyp.CENTER)
    indicator_spacing = np.linspace(xcenter-indicator_width*num_actions/2, xcenter+indicator_width*num_actions/2,num_actions)
    print indicator_spacing
    for i in range(num_actions):
        indicators.append(Indicator(indicator_spacing[i], height-50))
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

pyp.run()  #THIS DOES ALL THE ACTION