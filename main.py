import filter_env
from ddpg import *

import gc
gc.enable()

ENV_NAME = 'Pendulum-v0'
EPISODES = 150
TEST = 10
BASE_DIR = '/home/pcl/wang/upgraded/'
test_record = []
def main():
    origin_env = gym.make(ENV_NAME)
    env = filter_env.makeFilteredEnv(origin_env)
    agent = DDPG(env, 5)
    train_record = []
    # env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        print "episode:",episode
        # Train
        for step in xrange(200):
            origin_env.render()
            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            step = agent.perceive(state,action,reward,next_state,done) # t, actor_loss =
            if (step is not None) and step % 100 == 0:
                sum_weight = np.sum(agent.weights)
                # print sum_weight
                train_record.append(sum_weight)
            # if actor_loss is not None and t % 10 == 0:
                # train_record.append([t, actor_loss])
                # print t, actor_loss
            state = next_state
            if done:
                break

        # Testing:
        if episode % 5 == 0 and episode > 50:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(1000):
                    env.render()
                    action, _ = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            test_record.append([(episode-50)*200, ave_reward])
            print 'episode: ', episode, 'Evaluation Average Reward:', ave_reward
            np.save('actor_5_4.npy', np.array(test_record))

    for i in xrange(TEST):
        state = env.reset()
        usage_rate = np.zeros([20])
        state_trajectory = []
        u = []
        attention_weights = []
        for j in xrange(200):
            env.render()
            state_trajectory.append(state)
            action, weights = agent.action(state)  # direct action for test
            attention_weights.append(weights)
            u.append(action)
            state, reward, done, _ = env.step(action)
            usage_rate[agent.winner] += 1
            if done:
                break

        attention_weights = np.array(attention_weights)
        #np.savez(BASE_DIR + 'ctm_finaltest_' + str(i),
         #        usage_rate=usage_rate, states=np.array(state_trajectory),
          #       us=np.array(u), weights=agent.weights, attention_weights=np.array(attention_weights))

if __name__ == '__main__':
    main()
    # np.save('/home/pcl/wang/upgraded/noltm_ctm5.npy', np.array(test_record))
