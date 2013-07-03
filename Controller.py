
import numpy as np
from PhysicsModel import PhysicsModel
import matplotlib as mpl
import itertools
mpl.rcParams['image.interpolation'] = 'none'


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
            
            
        