import numpy as np
import random
from dqn_agent import Agent
import tensorflow as tf
# from DQN_agent_torch import Agent

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

import numpy as np

action_dict = {0: [0, 0, 0, 0],
               1: [0, 0, 0, 1],
               2: [0, 0, 1, 0],
               3: [0, 1, 0, 0],
               4: [1, 0, 0, 0],
               5: [0, 0, 1, 1],
               6: [0, 1, 0, 1],
               7: [1, 0, 0, 1],
               8: [0, 1, 1, 0],
               9: [1, 0, 1, 0],
               10: [1, 1, 0, 0],
               11: [0, 1, 1, 1],
               12: [1, 0, 1, 1],
               13: [1, 1, 1, 0],
               14: [1, 1, 0, 1],
               15: [1, 1, 1, 1], }


class Env:

  def __init__(self):
    self.pool = [i for i in range(10)]
    self.pool_sh = np.random.choice(self.pool, 6, replace=False)
    self.ans = np.random.choice(self.pool_sh, 4, replace=False)
    #self.ans = np.random.choice
    self.guess = np.zeros(4)
    self.done = False
    self.game_play = 40
    self.action_space = []
    for a in range(10):
        for b in range(10):
            for c in range(10):
                for d in range(10):
                    if a != b and a != c and a != d and b != c and b != d and c != d:
                        self.action_space.append([a, b, c, d])
    self.state = np.zeros(self.game_play*6)
    self.state_temp = np.zeros(6)
    self.count = 0
    self.A = 0
    self.B = 0

  # @tf.function
  def reset(self):
    self.count = 0
    self.state = np.zeros(self.game_play * 6)
    self.done = False
    self.ans = np.random.choice(self.pool_sh, 4, replace=False)
    self.guess = np.random.choice(self.pool, 4, replace=False)
    # self.guess = np.array([1, 2, 3, 4])
    self.A, self.B = self.analysis(self.guess, self.ans)
    self.state_temp = np.append(self.guess, [self.A, self.B])
    self.state = np.insert(np.delete(self.state.reshape(self.game_play, 6), 0, 0), 0, self.state_temp)



    return self.state







  @staticmethod
  def analysis(guess, ans):
    A = 0
    B = 0
    A = np.sum(guess == ans)
    B = np.sum(np.in1d(guess, ans))

    return A, B
  # @tf.function
  def step(self, action):
    self.count += 1
    # action = action_dict[action]
    # for i in range(len(action)):
    #   if action[i]> 0 :
    #       num = random.randint(0,9)
    #       if num not in self.guess:
    #         self.guess[i] = num
    self.guess = self.action_space[action]
    A, B = self.analysis(self.guess, self.ans)
    reward = (A-self.A)* 40 + (B-self.B)*40-4
    # reward =  + (B - self.B) * 40 - 8
    self.A = A
    self.B = B
    if A == 4 or self.count >= self.game_play-1:
      self.done = True
    self.state_temp = np.append(self.guess, [self.A, self.B])
    self.state = np.insert(np.delete(self.state.reshape(self.game_play, 6), self.count, 0), 6*self.count, self.state_temp)



    return self.state, reward, self.done

# def action_choice(action, guess):
#
#     for i in range(len(action)):
#         if action[i] == 0:
#             guess[i] = guess[i]
#         if action[i] == 1:
#             guess[i] = np.random.choice(guess)


# @tf.function
def f():
    if __name__ == '__main__':

      env = Env()
      episode = 0
      agent = Agent(lr=0.001, gamma=0.99, n_actions=5040, epsilon=1.0,
                    batch_size=5000, input_shape=[6*env.game_play],epsilon_dec=0.0001)
      n_games = 10000
      ddqn_scores = []
      eps_history = []
      best_score = 0
      filename = 'XAXB_ddqn.png'
      for i in range(n_games):
        done = False
        score = 0
        episode += 1
        observation = env.reset()
        while not done:
          action = agent.choose_action(observation)
          observation_, reward, done = env.step(action)
          score += reward
          agent.store_transition(observation, action, reward, observation_, done)
          observation = observation_
          agent.learn()

        eps_history.append(agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[-100:])
        print('episode: ', episode, 'score: %.2f' % score,
              ' average score %.2f' % avg_score, "count:", env.count, "A:", env.A ,"B:", env.B)

        if avg_score > best_score :
          best_score = avg_score
          print('Avg score %.2f better than best score %.2f' % (avg_score, best_score))
        if env.count < 20 and avg_score > best_score:
          agent.save_models()

      x = [i + 1 for i in range(n_games)]
      agent.plotLearning(x, ddqn_scores, eps_history, filename)

# with tf.device('/gpu:0'):
f()