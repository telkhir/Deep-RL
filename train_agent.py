import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from show_progress import ShowProgress
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.policies.policy_saver import PolicySaver
import logging
from tqdm import tqdm
import PIL
import os

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    # creating the environment
    max_episode_steps = 27000
    environment_name = "BreakoutNoFrameskip-v4"

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4])

    tf_env = TFPyEnvironment(env)

    # creating the Deep Q network
    preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, float) / 255.)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params)

    # creating the DQN agent
    train_step = tf.Variable(0)
    update_period = 4  # train model every 4 steps
    optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
    epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,  # initial epsilon
        decay_steps=250000,
        end_learning_rate=0.01)  # final epsilon
    agent = DqnAgent(tf_env.time_step_spec(),
                     tf_env.action_spec(),
                     q_network=q_net,
                     optimizer=optimizer,
                     target_update_period=2000,  # every 32,000 frames
                     td_errors_loss_fn=keras.losses.Huber(reduction="none"),  # must return error per instance
                     gamma=0.99,  # discount factor
                     train_step_counter=train_step,
                     epsilon_greedy=lambda: epsilon_fn(train_step))

    agent.initialize()

    # Create the reply buffer and the observer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000000)

    replay_buffer_observer = replay_buffer.add_batch

    # creating training metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]
    # creating the collect driver
    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=update_period)

    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                            tf_env.action_spec())
    initial_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer_observer, ShowProgress(20000)],
        num_steps=20000
    )
    final_time_step, final_policy_state = initial_driver.run()

    # creating the data set
    #trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=3)
    dataset = replay_buffer.as_dataset(sample_batch_size=64,
                                       num_steps=2,
                                       num_parallel_calls=3).prefetch(3)

    # creating the training loop (finally !!)
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    # Creating the training loop
    def train_agent(n_iterations):
        saver = PolicySaver(agent.policy, batch_size=tf_env.batch_size)
        time_step = None
        policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
        iterator = iter(dataset)
        for iteration in tqdm(range(n_iterations)):
            time_step, policy_state = collect_driver.run(time_step, policy_state)
            trajectories, buffer_info = next(iterator)
            train_loss = agent.train(trajectories)
            if iteration % 1000 == 0:
                print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")
                log_metrics(train_metrics)
            # save the policy each 10K iteration
            if iteration % 100000 == 0:
                saver.save('policy_%d' % iteration)

    #train the agent for 1M iteration, for debugging try small number of iteration ex 500
    train_agent(500)
