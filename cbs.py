from mapf_gym import dirDict, actionDict
import tensorflow as tf
from ACNet import ACNet
import numpy as np
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError


class GREEDY_SEARCH_WITH_CONSTRAINTS():
  '''
  This class provides functionality for running multiple instances of the 
  trained network in a single environment
  '''
  def __init__(self, model_path, grid_size, constraints = {}):
    '''
      constaints: {
        i: [((x,y), t)] # robot i should not be at position (x,y) at timestep t
      }
    '''
    self.grid_size = grid_size
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.network = ACNet("global",5,None,False,grid_size,"global")
    #load the weights from the checkpoint (only the global ones!)
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver = tf.train.Saver()
    saver.restore(self.sess,ckpt.model_checkpoint_path)
    self.constraints = constraints
      
  def set_env(self, gym):
    self.num_agents = gym.num_agents
    self.agent_states = []
    for i in range(self.num_agents):
        rnn_state = self.network.state_init
        self.agent_states.append(rnn_state)
    self.agent_positions, _, self.agent_goals = gym.world.scanForAgents()
    self.initial_world = gym.initial_world
      

  def next_position(self, agent, action):
    direction = dirDict[action]
    next_pos = (self.agent_positions[agent-1][0] + direction[0], 
                self.agent_positions[agent-1][1] + direction[1])
    return next_pos


  def is_position_valid(self, position):
    if position[0] < 0 \
       or position[0] >= self.initial_world.shape[0] \
       or position[1] < 0 \
       or position[1] >= self.initial_world.shape[1] \
       or self.initial_world[position[0], position[1]] == -1:
      return False
    return True


  def is_action_valid(self, agent, action):
    return self.is_position_valid(self.next_position(agent, action))

  def is_agent_constrained(self, agent, action, step):
    if agent in self.constraints:
      next_pos = self.next_position(agent, action)
      if (next_pos, step) in self.constraints[agent]:
        return True
    return False

  def step(self, agent, action):
    if(self.is_action_valid(agent, action)):
      self.agent_positions[agent - 1] = self.next_position(agent, action)
    else:
      raise Exception("action not valid in the state!")

  def completed(self):
    for i in range(self.num_agents):
      if self.agent_positions[i] != self.agent_goals[i]:
        return False
    return True

  def observe(self, agent):
    top_left = (self.agent_positions[agent-1][0]-self.grid_size//2, self.agent_positions[agent-1][1]-self.grid_size//2)
    bottom_right = (top_left[0] + self.grid_size, top_left[1] + self.grid_size)        
        
    observation_shape = (self.grid_size, self.grid_size)
    goal_map             = np.zeros(observation_shape)
    poss_map             = np.zeros(observation_shape)
    goals_map            = np.zeros(observation_shape)
    obs_map              = np.zeros(observation_shape)
    visible_agents = []
    for i in range(top_left[0],top_left[0]+self.grid_size):
      for j in range(top_left[1],top_left[1]+self.grid_size):
        if i>=self.initial_world.shape[0] or i<0 or j>=self.initial_world.shape[1] or j<0:
            #out of bounds, just treat as an obstacle
          obs_map[i-top_left[0],j-top_left[1]] = 1
          continue
        if self.initial_world[i,j] == -1:
          #obstacles
          obs_map[i - top_left[0], j - top_left[1]] = 1
        if self.agent_positions[agent-1] == (i,j):
          #agent's position
          poss_map[i - top_left[0], j - top_left[1]] = 1
        elif self.agent_goals[agent-1] == (i, j):
          #agent's goal
          goal_map[i - top_left[0], j - top_left[1]] = 1
        if (i, j) in self.agent_positions and self.agent_positions[agent-1] != (i, j):
          #other agents' positions
          visible_agent_id = self.agent_positions.index((i,j)) + 1
          visible_agents.append(visible_agent_id)
          poss_map[i - top_left[0], j - top_left[1]] = 1

    distance=lambda x1,y1,x2,y2:((x2-x1)**2+(y2-y1)**2)**.5

    for oth_agent in visible_agents:
      (x, y) = self.agent_goals[oth_agent - 1]
      if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
        #out of observation
        min_node=(-1,-1)
        min_dist=1000
        for i in range(top_left[0],top_left[0]+self.grid_size):
          for j in range(top_left[1],top_left[1]+self.grid_size):
            d=distance(i,j,x,y)
            if d<min_dist:
              min_node=(i,j)
              min_dist=d
        goals_map[min_node[0]-top_left[0],min_node[1]-top_left[1]]=1
      else:
        goals_map[x-top_left[0],y-top_left[1]]=1

    dx=self.agent_goals[agent-1][0]-self.agent_positions[agent-1][0]
    dy=self.agent_goals[agent-1][1]-self.agent_positions[agent-1][1]
    mag=(dx**2+dy**2)**.5
    if mag!=0:
      dx=dx/mag
      dy=dy/mag
    return ([poss_map,goal_map,goals_map,obs_map],[dx,dy,mag])

  def step_all_parallel(self, step):
    print(self.agent_positions)
    action_probs=[None for i in range(self.num_agents)]
    '''advances the state of the environment by a single step across all agents'''
    #parallel inference
    actions=[]
    inputs=[]
    goal_pos=[]
    for agent in range(1,self.num_agents+1):
        o = self.observe(agent)
        inputs.append(o[0])
        goal_pos.append(o[1])
    #compute up to LSTM in parallel
    h3_vec = self.sess.run([self.network.h3], 
                                     feed_dict={self.network.inputs:inputs,
                                                self.network.goal_pos:goal_pos})
    h3_vec=h3_vec[0]
    rnn_out=[]
    #now go all the way past the lstm sequentially feeding the rnn_state
    for a in range(0,self.num_agents):
        rnn_state=self.agent_states[a]
        lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                     feed_dict={self.network.inputs:[inputs[a]],
                                                self.network.h3:[h3_vec[a]],
                                                self.network.state_in[0]:rnn_state[0],
                                                self.network.state_in[1]:rnn_state[1]})
        rnn_out.append(lstm_output[0])
        self.agent_states[a]=state
    #now finish in parallel
    policy_vec=self.sess.run([self.network.policy], 
                                     feed_dict={self.network.rnn_out:rnn_out})
    policy_vec=policy_vec[0]
    for agent in range(1,self.num_agents+1):
        actions = np.flip(np.argsort(policy_vec[agent-1]))
        for action in actions:
          if(self.is_action_valid(agent, action) and not self.is_agent_constrained(agent, action, step)):
            self.step(agent, action)
            print(agent, action)
            break
    input()
      
  def find_path(self,max_step=256):
    '''run a full environment to completion, or until max_step steps'''
    solution=[]
    step=0
    while((not self.completed()) and step<max_step):
        timestep=[]
        for agent in range(self.num_agents):
            timestep.append(self.agent_positions[agent])
        solution.append(np.array(timestep))
        self.step_all_parallel(step + 1)
        step+=1
        #print(step)
    if step==max_step:
        raise OutOfTimeError
    for agent in range(self.num_agent):
        timestep.append(self.agent_positions[agent])
    return np.array(solution)

  # def step(self, (agent, action)):
  #   pass

  # def observe(self, agent):
  #   pass

  # def step_all_parallel(self):
  #   pass

  # def find_path(self):
  #   pass