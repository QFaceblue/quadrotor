import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
from copy import deepcopy
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境

# 超参数设置
ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e6   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward


class ActorModel(parl.Model):
    def __init__(self, act_dim):

        # 配置model结构

        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='tanh')
    def policy(self, obs):

        # 组装policy网络

        hid = self.fc1(obs)
        logits = self.fc2(hid)
        return logits


class CriticModel(parl.Model):
    def __init__(self):

        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)
    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q

class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

from parl.algorithms import DDPG # 也可以直接从parl库中快速引入DDPG算法，无需自己重新写算法

# class DDPG(parl.Algorithm):
#     def __init__(self,
#                  model,
#                  gamma=None,
#                  tau=None,
#                  actor_lr=None,
#                  critic_lr=None):
#         """  DDPG algorithm
        
#         Args:
#             model (parl.Model): actor and critic 的前向网络.
#                                 model 必须实现 get_actor_params() 方法.
#             gamma (float): reward的衰减因子.
#             tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
#             actor_lr (float): actor 的学习率
#             critic_lr (float): critic 的学习率
#         """
#         assert isinstance(gamma, float)
#         assert isinstance(tau, float)
#         assert isinstance(actor_lr, float)
#         assert isinstance(critic_lr, float)
#         self.gamma = gamma
#         self.tau = tau
#         self.actor_lr = actor_lr
#         self.critic_lr = critic_lr

#         self.model = model
#         self.target_model = deepcopy(model)

#     def predict(self, obs):
#         """ 使用 self.model 的 actor model 来预测动作
#         """
#         return self.model.policy(obs)

#     def learn(self, obs, action, reward, next_obs, terminal):
#         """ 用DDPG算法更新 actor 和 critic
#         """
#         actor_cost = self._actor_learn(obs)
#         critic_cost = self._critic_learn(obs, action, reward, next_obs,
#                                          terminal)
#         return actor_cost, critic_cost

#     def _actor_learn(self, obs):
#         action = self.model.policy(obs)
#         Q = self.model.value(obs, action)
#         cost = layers.reduce_mean(-1.0 * Q)
#         optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
#         optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
#         return cost

#     def _critic_learn(self, obs, action, reward, next_obs, terminal):
#         next_action = self.target_model.policy(next_obs)
#         next_Q = self.target_model.value(next_obs, next_action)

#         terminal = layers.cast(terminal, dtype='float32')
#         target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
#         target_Q.stop_gradient = True

#         Q = self.model.value(obs, action)
#         cost = layers.square_error_cost(Q, target_Q)
#         cost = layers.reduce_mean(cost)
#         optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
#         optimizer.minimize(cost)
#         return cost

#     def sync_target(self, decay=None, share_vars_parallel_executor=None):
#         """ self.target_model从self.model复制参数过来，可设置软更新参数
#         """
#         if decay is None:
#             decay = 1.0 - self.tau
#         self.model.sync_weights_to(
#             self.target_model,
#             decay=decay,
#             share_vars_parallel_executor=share_vars_parallel_executor)

# Agent负责算法与环境的交互，在交互过程中把生成的数据提供给Algorithm来更新模型(Model)，数据的预处理流程也一般定义在这里。
class QuadrotorAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QuadrotorAgent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        
        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        pre_action = action
        action = np.array([action[0]+0.1*action[1],action[0]+0.1*action[2],action[0]+0.1*action[3],action[0]+0.1*action[4]])
        action = np.clip(action, -1.0, 1.0)
        # print(action)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, pre_action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            # print("before:",action)
            action = np.squeeze(action)
            
            action = np.array([action[0]+0.1*action[1],action[0]+0.1*action[2],action[0]+0.1*action[3],action[0]+0.1*action[4]])
            # print(action)
            action = np.clip(action, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0], 
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1
            env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)

if __name__ == "__main__":
    # 创建飞行器环境
    # env = make_env("Quadrotor", task="hovering_control")
    env = make_env("Quadrotor", task="velocity_control", seed=0)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]
    # 更改输出维度
    act_dim = env.action_space.shape[0]+1

    # 根据parl框架构建agent
    model = QuadrotorModel(act_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm, obs_dim, act_dim)

    # # 加载模型
    # ckpt = 'model_dir/steps_680596_8423.2103163893.ckpt'  
    # agent.restore(ckpt)
    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

    # 启动训练
    # test_flag = 0
    # total_steps = 0
    # while total_steps < TRAIN_TOTAL_STEPS:
    #     train_reward, steps = run_episode(env, agent, rpm)
    #     total_steps += steps
    #     #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward)) # 打印训练reward

    #     if total_steps // TEST_EVERY_STEPS >= test_flag: # 每隔一定step数，评估一次模型
    #         while total_steps // TEST_EVERY_STEPS >= test_flag:
    #             test_flag += 1
    
    #         evaluate_reward = evaluate(env, agent)
    #         logger.info('Steps {}, Test reward: {}'.format(
    #             total_steps, evaluate_reward)) # 打印评估的reward

    #         # 每评估一次，就保存一次模型，以训练的step数命名
    #         ckpt = 'model_dir/steps_{}_{}.ckpt'.format(total_steps,evaluate_reward)
    #         agent.save(ckpt)
    # 评估
    ckpt = 'model_dir\steps_1000000_-20.866039639136154.ckpt'
    agent.restore(ckpt)
    evaluate_reward = evaluate(env, agent)
    logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward



