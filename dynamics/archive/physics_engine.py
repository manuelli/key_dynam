import os
import cv2
import numpy as np
import pymunk
from pymunk.vec2d import Vec2d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Polygon

from key_dynam.dynamics.utils import rand_float, rand_int, calc_dis, norm


class Engine(object):

    def __init__(self, dt, state_dim, action_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.param_dim = None
        self.state = None
        self.action = None
        self.param = None

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def get_action(self):
        return self.action.copy()

    def set_action(self, action):
        self.action = action.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


class BallActEngine(Engine):

    def __init__(self, dt, state_dim, action_dim):

        # state_dim = 4
        # action_dim = 2
        # param_dim = n_ball * (n_ball - 1)

        # param [relation_type, coefficient]
        # relation_type
        # 0 - no relation
        # 1 - spring (DampedSpring)
        # 2 - string (SlideJoint)
        # 3 - rod (PinJoint)

        super(BallActEngine, self).__init__(dt, state_dim, action_dim)

        self.init()

    def add_segments(self, p_range=(-80, 80, -80, 80)):
        a = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[0], p_range[3]), 1)
        b = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[1], p_range[2]), 1)
        c = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[0], p_range[3]), 1)
        d = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[1], p_range[2]), 1)
        a.friction = 1; a.elasticity = 1
        b.friction = 1; b.elasticity = 1
        c.friction = 1; c.elasticity = 1
        d.friction = 1; d.elasticity = 1
        self.space.add(a); self.space.add(b)
        self.space.add(c); self.space.add(d)

    def add_balls(self, center=(0., 0.), p_range=(-60, 60)):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        for i in range(self.n_ball):
            while True:
                x = rand_float(p_range[0], p_range[1])
                y = rand_float(p_range[0], p_range[1])
                flag = True
                for j in range(i):
                    if calc_dis([x, y], self.balls[j].position) < 25:
                        flag = False
                if flag:
                    break
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d((x, y))
            shape = pymunk.Circle(body, 0., (0, 0))
            shape.elasticity = 1
            self.space.add(body, shape)
            self.balls.append(body)

    def add_rels(self):
        param = np.zeros((self.n_ball * (self.n_ball - 1) // 2, 2))
        self.param_dim = param.shape[0]

        cnt = 0
        for i in range(self.n_ball):
            for j in range(i):
                # rel_type = rand_int(0, self.n_rel_type)
                rel_type = cnt % self.n_rel_type
                param[cnt, 0] = rel_type
                cnt += 1

                pos_i = self.balls[i].position
                pos_j = self.balls[j].position

                if rel_type == 0:
                    # no relation
                    pass

                elif rel_type == 1:
                    # spring
                    c = pymunk.DampedSpring(
                        self.balls[i], self.balls[j], (0, 0), (0, 0),
                        rest_length=60, stiffness=20, damping=0.)
                    self.space.add(c)

                elif rel_type == 2:
                    # rod
                    c = pymunk.PinJoint(
                        self.balls[i], self.balls[j], (0, 0), (0, 0))
                    self.space.add(c)

                else:
                    raise AssertionError("Unknown relation type")

        self.param = param

    def add_impulse(self, p_range=(-200, 200)):
        for i in range(self.n_ball):
            impulse = (rand_float(p_range[0], p_range[1]), rand_float(p_range[0], p_range[1]))
            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def add_boundary_impulse(self, p_range=(-70, 70, -70, 70)):
        f_scale = 2e2
        eps = 2
        for i in range(self.n_ball):
            impulse = np.zeros(2)
            p = np.array([self.balls[i].position[0], self.balls[i].position[1]])

            d = min(20, max(eps, p[0] - p_range[0]))
            impulse[0] += f_scale / d
            d = max(-20, min(-eps, p[0] - p_range[1]))
            impulse[0] += f_scale / d
            d = min(20, max(eps, p[1] - p_range[2]))
            impulse[1] += f_scale / d
            d = max(-20, min(-eps, p[1] - p_range[3]))
            impulse[1] += f_scale / d

            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def init(self, n_ball=5):
        self.space = pymunk.Space()
        self.space.gravity = (0., 0.)

        self.n_rel_type = 3
        self.n_ball = n_ball
        self.mass = 1.
        self.radius = 10
        self.balls = []
        # self.add_segments()
        self.add_balls()
        self.add_rels()
        self.add_impulse()

        self.state_prv = None

    @property
    def num_obj(self):
        return self.n_ball

    def get_state(self):
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    def add_action(self, action):
        if action is None:
            return
        self.balls[0].apply_force_at_local_point(force=action, point=(0, 0))

    def step(self, action=None):
        self.state_prv = self.get_state()
        self.add_action(action)
        self.add_boundary_impulse()
        self.space.step(self.dt)

    def render(self, states, actions, param, video=True, image=False, path=None, draw_edge=True,
               lim=(-80, 80, -80, 80), verbose=True):
        # states: time_step x n_ball x 4
        # actions: time_step x 2

        # lim = (lim[0] - self.radius, lim[1] + self.radius, lim[2] - self.radius, lim[3] + self.radius)

        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            if verbose:
                print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

        if image:
            image_path = path
            if verbose:
                print('Save images to %s' % image_path)
            command = 'mkdir -p %s' % image_path
            os.system(command)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'lightsteelblue']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
            plt.xlim(lim[0], lim[1])
            plt.ylim(lim[2], lim[3])
            # plt.axis('off')

            fig.set_size_inches(1.5, 1.5)

            if draw_edge:
                # draw force
                F = actions[i, 0]

                normF = norm(F)
                Fx = F / normF * normF * 0.05
                st = states[i, 0, :2] + F / normF * 12.
                ax.arrow(st[0], st[1], Fx[0], Fx[1], fc='Orange', ec='Orange', width=3., head_width=15., head_length=15.)

                # draw edge
                cnt = 0
                for x in range(n_ball):
                    for y in range(x):
                        rel_type = int(param[cnt, 0]); cnt += 1
                        if rel_type == 0:
                            continue

                        plt.plot([states[i, x, 0], states[i, y, 0]],
                                 [states[i, x, 1], states[i, y, 1]],
                                 '-', color=c[rel_type], lw=2, alpha=0.5)

            circles = []
            circles_color = []
            for j in range(n_ball):
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius)
                circles.append(circle)
                circles_color.append(c[j % len(c)])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=0.5)
            ax.add_collection(pc)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.tight_layout()

            if video or image:
                fig.canvas.draw()
                frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = frame[21:-19, 21:-19]

            if video:
                out.write(frame)
                if i == time_step - 1:
                    for _ in range(5):
                        out.write(frame)

            if image:
                cv2.imwrite(os.path.join(image_path, 'fig_%s.png' % i), frame)

            plt.close()

        if video:
            out.release()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='')
    args = parser.parse_args()

    os.system('mkdir -p test')

    if args.env == 'BallAct':
        dt = 1. / 50.

        n_ball = 5
        state_dim = 4
        action_dim = 2
        engine = BallActEngine(dt, state_dim, action_dim)

        time_step = 500
        states = np.zeros((time_step, n_ball, 4))
        actions = np.zeros((time_step, 1, 2))

        act = np.zeros(2)
        for i in range(time_step):
            states[i] = engine.get_state()
            act += (np.random.rand(2) - 0.5) * 600 - act * 0.1 - states[i, 0, 2:] * 0.1
            act = np.clip(act, -1000, 1000)
            engine.step(act)
            actions[i, 0] = act

        engine.render(states, actions, engine.get_param(), video=True, image=True, path='test/BallAct')
        engine.render(states, actions, engine.get_param(), video=True, image=True, path='test/BallAct_inv', draw_edge=False)

