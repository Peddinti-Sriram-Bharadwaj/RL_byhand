#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import time

matplotlib.use('Agg')

# Ensure the directory exists
os.makedirs('./bandit', exist_ok=True)

# Optimize NumPy for Apple Silicon
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count())
os.environ["ACCELERATE_ENABLE_AUTO_PARALLELIZATION"] = "1"
os.environ["ACCELERATE_NUM_THREADS"] = str(cpu_count())

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k, dtype=np.int32)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        # Vectorized initialization with float32 for better performance
        self.q_true = np.random.randn(self.k).astype(np.float32) + self.true_reward
        self.q_estimation = np.full(self.k, self.initial, dtype=np.float32)
        self.action_count = np.zeros(self.k, dtype=np.int32)
        self.best_action = int(np.argmax(self.q_true))
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.indices))

        if self.UCB_param is not None:
            # Vectorized UCB calculation
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            return int(np.random.choice(np.where(UCB_estimation == np.max(UCB_estimation))[0]))

        if self.gradient:
            # Numerical stability for exp calculation
            exp_est = np.exp(self.q_estimation - np.max(self.q_estimation))
            self.action_prob = exp_est / np.sum(exp_est)
            return int(np.random.choice(self.indices, p=self.action_prob))

        # Vectorized max operation
        return int(np.random.choice(np.where(self.q_estimation == np.max(self.q_estimation))[0]))

    def step(self, action):
        # Generate reward using float32 for better performance
        reward = np.random.randn() + self.q_true[action] ## simulating getting answer from the environment as a normal distribution of some true value
        self.time += 1 
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time # moving average technique to update the reward. 
        
        
        ## select one way to update the action values. 
        if self.sample_averages:
            # Vectorized update
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k, dtype=np.float32)
            one_hot[action] = 1
            baseline = self.average_reward if self.gradient_baseline else 0
            # Vectorized gradient update
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # Vectorized constant step size update
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def run_simulation(args):
    bandit, run_id, time = args
    rewards = np.zeros(time, dtype=np.float32)
    best_action_counts = np.zeros(time, dtype=np.float32)
    
    bandit.reset()
    for t in range(time):
        action = bandit.act()
        reward = bandit.step(action)
        rewards[t] = reward
        best_action_counts[t] = 1 if action == bandit.best_action else 0
        
    return rewards, best_action_counts

def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time), dtype=np.float32)
    best_action_counts = np.zeros_like(rewards)
    
    # Use all available cores for parallel processing
    with Pool(cpu_count()) as pool:
        for i, bandit in enumerate(bandits):
            # Create argument list for parallel processing
            args = [(bandit, r, time) for r in range(runs)]
            
            # Run simulations in parallel
            results = list(tqdm(
                pool.imap(run_simulation, args),
                total=runs,
                desc=f'Bandit {i+1}/{len(bandits)}'
            ))
            
            # Unpack results
            for r, (reward, best_action) in enumerate(results):
                rewards[i, r] = reward
                best_action_counts[i, r] = best_action

    return best_action_counts.mean(axis=1), rewards.mean(axis=1)

def figure_2_1():
    # Use float32 for better performance
    data = np.random.randn(200, 10).astype(np.float32) + np.random.randn(10).astype(np.float32)
    plt.violinplot(dataset=data)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('./bandit/figure_2_1.png')
    plt.close()

def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='$\\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='$\\epsilon = %.02f$' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./bandit/figure_2_2.png')
    plt.close()

def figure_2_3(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='$\\epsilon = 0, q = 5$')
    plt.plot(best_action_counts[1], label='$\\epsilon = 0.1, q = 0$')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('./bandit/figure_2_3.png')
    plt.close()

def figure_2_4(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB $c = 2$')
    plt.plot(average_rewards[1], label='epsilon greedy $\\epsilon = 0.1$')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('./bandit/figure_2_4.png')
    plt.close()

def figure_2_5(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('./bandit/figure_2_5.png')
    plt.close()

def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float32),
                  np.arange(-5, 2, dtype=np.float32),
                  np.arange(-4, 3, dtype=np.float32),
                  np.arange(-2, 3, dtype=np.float32)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('./bandit/figure_2_6.png')
    plt.close()

if __name__ == '__main__':
    start_time = time.time()  # Record the start time

    # Run the figures
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate total execution time
    print(f"Total execution time: {total_time:.2f} seconds")