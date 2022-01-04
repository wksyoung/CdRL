# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from identifier_network import IdentifierNetwork
from replay_buffer import ReplayBuffer
from ctm_utils import uptree, sleeping_experts
from reward_identifier import RewardNetwork

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000# 10000
BATCH_SIZE = 64
GAMMA = 0.99
H = 3

class DDPG:
    """docstring for DDPG"""
    def __init__(self, env, actor_n):
        self.name = 'DDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, actor_n)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)
        self.identifier = IdentifierNetwork(self.sess, self.state_dim, self.action_dim)
        self.reward_network1 = RewardNetwork(self.sess,self.state_dim, self.action_dim)
        self.reward_network2 = RewardNetwork(self.sess, self.state_dim, self.action_dim)

        self.actor_num = actor_n
        self.weights = np.ones([actor_n])
        self.winner = None
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)
        self.train_step = 1

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])
        # train identifier network
        reward_batch = np.resize(reward_batch, [BATCH_SIZE, 1])
        self.identifier.update(state_batch, action_batch, next_state_batch)
        if self.train_step % 2 == 0:
            self.reward_network1.update(state_batch, action_batch, reward_batch)
        else:
            self.reward_network2.update(state_batch, action_batch, reward_batch)
        self.train_step += 1

        # Calculate y_batch
        action_votes = [self.actor_network.actions(next_state_batch, i) for i in range(self.actor_num)]
        state_votes = [next_state_batch for i in range(self.actor_num)]
        reward_votes = np.array([reward_batch for i in range(self.actor_num * 2)])
        targets_mean = []
        targets_weight = []
        for step in range(1, H):
            r1 = [self.reward_network1.predict_batch(state_votes[i], action_votes[i]) for i in range(self.actor_num)]
            r2 = [self.reward_network2.predict_batch(state_votes[i], action_votes[i]) for i in range(self.actor_num)]
            rvs = np.array(r1 + r2) * (GAMMA ** step)
            reward_votes = reward_votes + rvs  # The 80 reward votes
            state_votes = [self.identifier.predict_batch(state_votes[i], action_votes[i]) for i in range(self.actor_num)]
            action_votes = [self.actor_network.actions(state_votes[i], i) for i in range(self.actor_num)]
            target_actions = [self.actor_network.target_actions(state_votes[i], self.winner) for i in range(self.actor_num)]
            target_qs = [self.critic_network.target_q(state_votes[i],target_actions[i]) for i in range(self.actor_num)]
            targets_one_step = reward_votes + (GAMMA ** (step + 1)) * np.array(target_qs + target_qs)
            targets_mean.append(np.mean(targets_one_step, 0))
            targets_weight.append(1 / np.var(targets_one_step, 0))

        targets_mean = np.array(targets_mean)
        # q_value_batch = np.mean(targets_mean, 0)
        targets_weight = np.array(targets_weight)
        q_value_batch = np.sum(np.multiply(targets_mean, targets_weight / np.sum(targets_weight, 0)), 0)
        # next_action_batch = self.actor_network.target_actions(next_state_batch,self.winner)
        # q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch, self.winner)
        # actor_loss = -tf.reduce_mean(self.critic_network.q_value(state_batch, action_batch_for_gradients))
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(self.winner, state_batch, q_gradient_batch)

        # Update the target networks
        self.actor_network.update_target(self.winner)
        self.critic_network.update_target()
        return self.train_step# , self.sess.run(actor_loss)

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action, _ = self.action(state)
        return action+self.exploration_noise.noise()

    def action(self, state):
        if self.winner is not None and self.replay_buffer.count() > REPLAY_START_SIZE + 2:
            action_candidates = np.array([self.actor_network.action(state, i) for i in range(self.actor_num)])
            state_batch = np.array([state for i in range(self.actor_num)])
            scores = np.reshape(self.critic_network.q_value(state_batch, action_candidates), [self.actor_num,])
            scores -= np.min(scores)
        else:
            scores = np.ones([self.actor_num])
        cp = scores * self.weights
        winner_id = uptree(cp)
        self.winner = winner_id
        action = self.actor_network.action(state, winner_id)
        return action, cp

    def perceive(self,state,action,reward,next_state,done):
        step=None
        # q_loss = None
        # actor_loss = None
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            step = self.train()
            sleeping_experts(state, action, self.weights, self.actor_network, self.identifier)
        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()
        return step









