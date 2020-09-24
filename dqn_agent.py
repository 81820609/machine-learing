from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""Model的另一種建立方法!!"""
'''
inputs = keras.Input(shape=(28, 28, 1,))
x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.summary()

"""也能創立一個sub_class!!"""
'''
'''
class DeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc_layer_dims):
        super(DeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc_layer_dims[0], activation='relu')
        self.dense2 = keras.layers.Dense(fc_layer_dims[1], activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        # V = self.V(x)
        A = self.A(x)

        # Q = (V + (A - tf.math.reduce_mean(A, axis=0, keepdims=True)))

        return A
'''

# @tf.function
def build_dqn(lr, input_dims, fc1_dims, fc2_dims, n_actions):
    model = Sequential()
    model.add(Dense(fc1_dims, input_shape=[*input_dims, ],
                    activation='relu'))
    model.add(Dense(fc2_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return model


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_momery = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # @tf.function
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_momery[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # @tf.function
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_momery[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


class Agent:

    def __init__(self, lr, gamma, epsilon, batch_size, n_actions, input_shape,
                 epsilon_dec=1e-3, epsilon_end=0.01, mem_size=50000, replace=100,
                 fc_layer_dims=(64, 64), q_eval_fname='q_eval.h5',
                 q_target_fname='q_target.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.replace = replace
        self.batch_size = batch_size
        self.q_eval_file = q_eval_fname
        self.q_target_file = q_target_fname

        self.learning_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_shape)
        self.q_eval = build_dqn(lr, input_shape, fc_layer_dims[0],
                              fc_layer_dims[1], n_actions)
        # self.q_eval = tf.convert_to_tensor(self.q_eval)
        self.q_next = build_dqn(lr, input_shape, fc_layer_dims[0],
                              fc_layer_dims[1], n_actions)
        # self.q_next = tf.convert_to_tensor(self.q_next)
        # self.q_eval = DeepQNetwork(n_actions, fc_layer_dims)
        # self.q_next = DeepQNetwork(n_actions, fc_layer_dims)

        # self.q_eval.compile(optimizer=Adam(learning_rate=lr),
        #                     loss='mean_squared_error')
        # self.q_next.compile(optimizer=Adam(learning_rate=lr),
        #                     loss='mean_squared_error')\

    # @tf.function
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # @tf.function
    def choose_action(self, observation_space):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation_space], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    @tf.function
    def replace_target_network(self):
        if self.replace != 0 and self.learning_step_counter % self.replace == 0:
            # self.q_next.set_weights(self.q_eval.get_weights())
            # print(self.q_next.weights)
            self.q_next.weights[0] = self.q_eval.weights[0]
            # print( self.q_next.get_weights())
            # print(self.q_next.weights[0])
    # @tf.function
    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        self.replace_target_network()

        # self.q_eval = self.q_eval.eval()
        # self.q_next = self.q_next.eval()

        q_eval = self.q_eval.predict(states)
        q_next = self.q_next.predict(new_states)

        q_next[dones] = 0.0

        q_target = q_eval[:]

        indices = np.arange(self.batch_size)
        q_target[indices, actions] = rewards + \
                                     self.gamma * np.max(q_next, axis=1)

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > \
                                                          self.epsilon_min else self.epsilon_min

        self.learning_step_counter += 1
        # print(self.learning_step_counter)



    def save_models(self):
        self.q_eval.save(self.q_eval_file)
        self.q_next.save(self.q_target_file)
        print("... saving models ...")

    def load_models(self):
        self.q_eval = load_model(self.q_eval_file)
        self.q_next = load_model(self.q_target_file)

    def plotLearning(self, x, scores, epsilons, filename, lines=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)

        ax.plot(x, epsilons, color="C0")
        ax.set_xlabel("Game", color="C0")
        ax.set_ylabel("Epsilon", color="C0")
        ax.tick_params(axis='x', colors="C0")
        ax.tick_params(axis='y', colors="C0")

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

        ax2.scatter(x, running_avg, color="C1")
        # ax2.xaxis.tick_top()
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        # ax2.set_xlabel('x label 2', color="C1")
        ax2.set_ylabel('Score', color="C1")
        # ax2.xaxis.set_label_position('top')
        ax2.yaxis.set_label_position('right')
        # ax2.tick_params(axis='x', colors="C1")
        ax2.tick_params(axis='y', colors="C1")

        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(filename)