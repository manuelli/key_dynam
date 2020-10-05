from __future__ import print_function

# Python imports
from gym import Env
import numpy as np
import matplotlib.pyplot as plt

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *
import pygame.surfarray

# pymunk imports
import pymunk
import pymunk.pygame_util
import pymunk.matplotlib_util


# key_dynam imports
# from key_dynam.envs.task import Task
from key_dynam.envs.utils import get_body_state_as_dict, set_body_state_from_dict, zero_body_velocity

WHITE = [255,255,255]
BLUE = [0,0,255]
RED = [255,0,0]

class PusherSlider(Env):

    def __init__(self, config):
        super(Env, self)
        pygame.init()

        self._space = pymunk.Space()
        self.config = config
        self._sim_time = 0.0 # sim time, incremented manually when 'step' is called
        self._space_has_been_stepped = False

    @property
    def config(self):
        """
        Dict containing parameters
        :return:
        :rtype:
        """
        return self._config

    @config.setter
    def config(self, val):
        self._config = val

    def setup_environment(self):
        """
        Creates the pymunk environment, adds the objects etc.
        :return:
        :rtype:
        """
        env_config = self.config["env"]

        self._space = pymunk.Space()
        self._space.gravity = env_config["gravity"]

        # Physics
        # Time step
        self._dt = env_config["dt"]

        # Number of physics steps per screen frame
        self._physics_steps_per_frame = env_config["physics_steps_per_frame"]
        self._fps = 1.0/(self._dt * self._physics_steps_per_frame)

        # pygame
        pygame.init()
        self._pygame_render_initialized = False
        self._mpl_render_initialized = False
        self._pygame_clock = pygame.time.Clock()


        # Static barrier walls (lines) that the balls bounce off of
        self._add_slider(self.config)
        self._add_pusher(self.config)

        # Execution control and ,time until the next ball spawns
        self._running = True

    @property
    def fps(self):
        return self._fps

    # gym functions
    def step(self, action):
        """
        OpenAI gym function

        :param action: 2-element vector that encoders pusher velocity, np.vector with
        shape [2,]
        :type action:
        :return: observation, reward, done, info
        :rtype:
        """

        # step the simulation
        for i in range(self.config["env"]["physics_steps_per_frame"]):
            self._pusher_body.velocity = action
            self._space.step(self.config["env"]["dt"])
            self._sim_time += self.config["env"]["dt"]

            self._space_has_been_stepped = True

        observation = self.get_observation()
        reward = self.get_reward()
        done = False
        info = dict() # contains useful additional information

        return observation, reward, done, info

    def get_observation(self):
        """
        Returns the observation, including images
        :return:
        :rtype: dict()
        """

        obs = dict()
        obs["pusher"] = get_body_state_as_dict(self._pusher_body)
        obs["slider"] = get_body_state_as_dict(self._slider_body)
        obs['sim_time'] = self._sim_time

        if "observation" in self.config["env"]: # for backwards compatibility
            # space must have been stepped for this to work
            if not self._space_has_been_stepped:
                raise ValueError("you must call step() before using this method")
            if self.config["env"]["observation"]["rgb_image"]:
                rgb_array = self.render(mode="rgb_array")
                obs["image"] = dict()
                obs["image"]["camera_0"] = {"rgb": rgb_array}
        else:
            obs["image"] = None

        return obs

    def get_reward(self):
        return 0.

    def get_info(self):
        """
        Gets info field for OpenAI gym API
        :return:
        :rtype:
        """
        return None

    def reset(self):
        """
        Sets up the environment according to the config
        :return:
        :rtype:
        """
        self._pygame_render_initialized = False
        self._mpl_render_initialized = False
        self._space_has_been_stepped = False

        self.setup_environment()
        self.set_initial_positions()
        self._sim_time = 0.

    def render(self,
               mode='human',
               fps=None, # max FPS for pygame rendering
               ):

        if not self._pygame_render_initialized:
            self._initialize_pygame_render()

        self._pygame_clear_screen()
        self._pygame_draw_objects()
        pygame.display.flip()  # what does this do?
        # Delay fixed time between frames

        if mode =='human':
            if fps is None:
                fps = self._fps
            self._pygame_clock.tick(fps)
            pygame.display.set_caption("fps: " + str(self._pygame_clock.get_fps()))
        elif mode == "rgb_array":
            rgb_array = self.get_image_as_rgb_array()
            return rgb_array
        else:
            raise ValueError("unrecognized render mode: %s" %(mode))

    # end gym functions

    def get_image_as_rgb_array(self):
        """
        Returns H x W x 3 numpy array
        :return:
        :rtype:
        """
        return pygame.surfarray.array3d(self._pygame_screen).swapaxes(0,1)


    def _initialize_pygame_render(self):
        self._pygame_render_initialized = True
        self._pygame_screen = pygame.display.set_mode(self.config["env"]["display_size"])
        self._pygame_draw_options = pymunk.pygame_util.DrawOptions(self._pygame_screen)
        self._pygame_clock = pygame.time.Clock()

    def process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """

        velocity = 100
        action = np.array([0,0])

        for event in pygame.event.get():

            # generic events about quitting out of the game etc.
            if event.type == QUIT:
                self._running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._running = False

        pressed = pygame.key.get_pressed()
        if pressed[K_LEFT]:
            action += np.array([-velocity, 0])

        if pressed[K_RIGHT]:
            action += np.array([velocity, 0])

        if pressed[K_UP]:
            action += np.array([0, velocity])

        if pressed[K_DOWN]:
            action += np.array([0, -velocity])

        return action

    def _add_slider(self, global_config):
        """
        Creates the slider object
        :return:
        :rtype:
        """

        config = global_config["env"]["slider"]
        mass = config["mass"]

        damping_override = config["damping"]

        body = None
        shape = None
        if config["type"] == "Box":
            size = config["size"]
            inertia = pymunk.moment_for_box(mass, size)
            body = pymunk.Body(mass, inertia)
            shape = pymunk.Poly.create_box(body, size)
        elif config["type"] == "Circle":
            # following example in Body.__init__ documentation section
            # http://www.pymunk.org/en/latest/pymunk.html#pymunk.Body
            body = pymunk.Body()
            radius = config["size"]
            shape = pymunk.Circle(body, radius)
            shape.mass = mass # sets the mass
        else:
            raise ValueError("unsupported slider type: %s" %(config["type"]))


        if "friction" in config:
            shape.friction = config["friction"]
        else:
            shape.friction = 0.9

        shape.color = config["color"]

        def velocity_func(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping_override, dt)

        body.velocity_func = velocity_func
        self._slider_body = body
        self._slider_shape = shape
        self._space.add(body, shape)


    def _add_pusher(self, global_config):
        """
        Creates the pusher object
        :param global_config:
        :type global_config:
        :return:
        :rtype:
        """

        config = global_config["env"]["pusher"]
        mass = config["mass"]
        radius = config["radius"]
        body = pymunk.Body(mass, pymunk.inf)
        shape = pymunk.Circle(body, radius)
        shape.color = config["color"]

        # elasiticity and collision type

        self._space.add(body, shape)
        self._pusher_body = body
        self._pusher_shape = shape

    def set_initial_positions(self):
        """
        Moves the bodies to their starting positions
        :return:
        :rtype:
        """
        self._slider_body.position = (300, 300)
        self._pusher_body.position = (200, 300)

    def _pygame_clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self._pygame_screen.fill(WHITE)

    def _pygame_draw_objects(self):
        """
        Draw the objects in pygame window
        :return: None
        """
        self._space.debug_draw(self._pygame_draw_options)

    def set_object_states(self, state_dict):
        """
        Sets body positions/velocities given state dict
        :param state_dict:
        :type state_dict:
        :return:
        :rtype:
        """
        set_body_state_from_dict(self._pusher_body, state_dict['pusher'])
        set_body_state_from_dict(self._slider_body, state_dict['slider'])

    def set_object_positions_for_rendering(self, state_dict):
        """
        Sets positions but sets velocities to zero. Call self._space.step() to
        update the state
        :param state_dict:
        :type state_dict:
        :return:
        :rtype:
        """
        set_body_state_from_dict(self._pusher_body, state_dict['pusher'])
        zero_body_velocity(self._pusher_body)

        set_body_state_from_dict(self._slider_body, state_dict['slider'])
        zero_body_velocity(self._slider_body)

        self.step_space_to_initialize_render()

    def step_space_to_initialize_render(self):
        """
        If you don't do this then the space doesn't update and the rendering
        will be incorrect.
        :return:
        :rtype:
        """
        eps = 1e-5
        self._space.step(eps)

    @property
    def pusher_body(self):
        return self._pusher_body

    @property
    def slider_body(self):
        return self._slider_body

    @staticmethod
    def make_default_config():
        config = dict()

        slider = dict()
        slider["mass"] = 10
        slider["size"] = (50, 100)
        slider["friction"] = 0.9
        slider["color"] = BLUE
        slider["damping"] = 0.05

        pusher = dict()
        pusher["mass"] = 500
        pusher["radius"] = 5
        pusher["color"] = RED


        config["pusher"] = pusher
        config["slider"] = slider
        config["dt"] = 1./60
        config["gravity"] = (0.0, 0.)
        config["physics_steps_per_frame"] = 1
        config["display_size"] = (600, 600)
        config["type"] = "PusherSlider"

        config["observation"] = dict()

        # whether or not to include rgb_image in the observation
        config["observation"]["rgb_image"] = False

        global_config = dict()
        global_config['env'] = config

        # whether or not to include rgb_image in the observation
        global_config["observation"] = dict()
        global_config["observation"]["rgb_image"] = False

        return global_config

    @staticmethod
    def make_default():
        config = PusherSlider.make_default_config()
        env = PusherSlider(config)
        return env


    @staticmethod
    def box_keypoints(width,  # float
                      height,  # float
                      ): # return -> np.array shape [N, 2], N is num keypoints
        N = 4
        pts = np.zeros([N,2])

        width_half = width/2.0
        height_half = height/2.0

        # bottom left corner
        pts[0,:] = np.array([-width_half, -height_half])

        # top left corner
        pts[1,:] = np.array([-width_half, height_half])

        # top right corner
        pts[2,:] = np.array([width_half, height_half])

        # bottom right corner
        pts[3, :] = np.array([width_half, -height_half])

        return pts



