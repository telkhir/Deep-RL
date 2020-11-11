import PIL
import os
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from show_progress import ShowProgress
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.utils.common import function
# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

if __name__ == '__main__':

    # creating the environment
    max_episode_steps = 27000
    environment_name = "BreakoutNoFrameskip-v4"

    env = suite_atari.load(
        environment_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4])

    tf_env = TFPyEnvironment(env)

    frames = []


    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    def plot_animation(frames, repeat=False, interval=40):
        fig = plt.figure()
        patch = plt.imshow(frames[0])
        plt.axis('off')
        anim = animation.FuncAnimation(
            fig, update_scene, fargs=(frames, patch),
            frames=len(frames), repeat=repeat, interval=interval)
        plt.close()
        return anim

    def save_frames(trajectory):
        global frames
        frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

    prev_lives = tf_env.pyenv.envs[0].ale.lives()

    def reset_and_fire_on_life_lost(trajectory):
        global prev_lives
        lives = tf_env.pyenv.envs[0].ale.lives()
        if prev_lives != lives:
            tf_env.reset()
            tf_env.pyenv.envs[0].step(1)
            prev_lives = lives

    # load the policy
    #saved_policy = tf.saved_model.load("policy_0")
    saved_policy = tf.compat.v2.saved_model.load("policy_100000")
    #policy_state = saved_policy.get_initial_state(batch_size=3)
    print(saved_policy)

    watch_driver = DynamicStepDriver(
        tf_env,
        saved_policy,
        observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
        num_steps=1000)

    #watch_driver.run = function(watch_driver.run)
    time_step = None
    policy_state = saved_policy.get_initial_state()
    final_time_step, final_policy_state = watch_driver.run(time_step, policy_state)

    plot_animation(frames)


    # generate gif

    image_path = os.path.join("images", "rl", "breakout.gif")
    frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
    frame_images[0].save(image_path, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=30,
                         loop=0)