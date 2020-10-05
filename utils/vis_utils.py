import matplotlib.pyplot as plt
import cv2
import numpy as np


class ImageVisualizer():

    def __init__(self, num_rows, num_cols, name="Image Visualizer",
                 interactive=True):

        if interactive:
            plt.ion()

        # width
        figsize = (6.4 * num_cols, 4.8 * num_rows)
        self._fig = plt.figure(name, figsize=figsize)
        self._axes = self._fig.subplots(num_rows, num_cols, squeeze=False)
        self._artists = [[None for i in range(num_cols)] for i in range(num_rows)]
        self._flag = False

    @property
    def fig(self):
        return self._fig

    def draw_image(self, row_idx, col_idx, img):
        artist = self._artists[row_idx][col_idx]
        ax = self._axes[row_idx, col_idx]

        if artist is None:
            self._artists[row_idx][col_idx] = ax.imshow(img)
        else:
            artist.set_data(img)

        # ax.imshow(img)
        # ax.set_data(img)

    def visualize_interactive(self):
        # hacky solution copied from
        # this is to avoid it stealing focus all the time
        # https://github.com/matplotlib/matplotlib/issues/11131#issuecomment-563505937

        # needed to get it to render the first time through the loop
        if not self._flag:
            plt.pause(0.001)
            self._flag = True

        self._fig.canvas.draw_idle()
        self._fig.canvas.start_event_loop(0.001)


def create_video(image_filenames,
                 save_filename,
                 fps=15
                 ):
    img_array = []
    for filename in image_filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print("height", height)
    print("width", width)
    out = cv2.VideoWriter(save_filename,
                          cv2.VideoWriter_fourcc(*'MP4V'),
                          fps,
                          size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def draw_keypoints(img,
                   u_vec,
                   v_vec,
                   size=None
                   ):
    cm = plt.get_cmap('gist_rainbow')

    if size is None:
        size = int(0.02 * img.shape[0])

    # draws multiple reticles
    n = len(u_vec)
    for i in range(n):
        u = u_vec[i]
        v = v_vec[i]

        color = 255*np.array(cm(1. * i / n))[:3]

        # u and v might be flipped here
        cv2.circle(img, (u, v), size, color, -1)
