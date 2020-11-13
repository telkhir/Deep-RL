import base64
import imageio
import IPython
import PIL
import PIL.Image
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from tf_agents.environments.tf_py_environment import TFPyEnvironment
# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')


class RenderingUtils:

    def __init__(self, frames):
        self.frames = frames

    def update_scene(self, num, patch):
        patch.set_data(self.frames[num])
        return patch,

    def plot_animation(self, repeat=False, interval=40):
        fig = plt.figure()
        patch = plt.imshow(self.frames[0])
        plt.axis('off')
        anim = animation.FuncAnimation(
            fig, self.update_scene, fargs=(self.frames, patch),
            frames=len(self.frames), repeat=repeat, interval=interval)
        plt.close()
        return anim

    # Generate gif
    def generate_gif(self, path):
        image_path = os.path.join(path)
        frame_images = [PIL.Image.fromarray(frame) for frame in self.frames[:150]]
        frame_images[0].save(image_path, format='GIF',
                             append_images=frame_images[1:],
                             save_all=True,
                             duration=30,
                             loop=0)

    # Generate video
    @staticmethod
    def embed_mp4(filename):
        """Embeds an mp4 file in the notebook."""
        video = open(filename, 'rb').read()
        b64 = base64.b64encode(video)
        tag = '''
          <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4">
          Your browser does not support the video tag.
          </video>'''.format(b64.decode())

        return IPython.display.HTML(tag)

    def create_policy_eval_video(self, env, policy, filename, num_episodes=5, fps=30):
        filename = filename + ".mp4"
        tf_env = TFPyEnvironment(env)
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(num_episodes):
                time_step = tf_env.reset()
                tf_env.step(1)
                video.append_data(env.render())
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = tf_env.step(action_step.action)
                    video.append_data(env.render())
            video.close()
        return self.embed_mp4(filename)
