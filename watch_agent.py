import tensorflow as tf
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from show_progress import ShowProgress
from output_util import RenderingUtils

if __name__ == '__main__':

    # creating the environment
    max_episode_steps = 27000
    environment_name = "BreakoutNoFrameskip-v4"

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4])

    tf_env = TFPyEnvironment(env)

    # load the policy : load the last saved policy
    saved_policy = tf.compat.v2.saved_model.load("policy_100000")

    frames = []

    def save_frames(trajectory):
        global frames
        frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

    prev_lives = tf_env.pyenv.envs[0].ale.lives()

    def reset_and_fire_on_life_lost(trajectory):
        global prev_lives
        lives = tf_env.pyenv.envs[0].ale.lives()
        if prev_lives != lives:
            tf_env.reset()
            tf_env.step(1)
            prev_lives = lives


    watch_driver = DynamicStepDriver(
        tf_env,
        saved_policy,
        observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
        num_steps=1000)

    tf_env.reset()  # reset the env
    time_step = tf_env.step(1)  # fire the ball to begin playing
    policy_state = saved_policy.get_initial_state()  # empty state ()
    final_time_step, final_policy_state = watch_driver.run(time_step, policy_state)

    # render a window that shows the agent plays (works on the jupyter notebook)
    renderingUtils = RenderingUtils(frames)

    renderingUtils.plot_animation()

    renderingUtils.generate_gif("breakout.gif")

    renderingUtils.create_policy_eval_video(env, saved_policy, "trained-agent")