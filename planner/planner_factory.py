import os
import numpy as np

from key_dynam.planner.planners import PlannerMPPI


def planner_from_config(config):
    planner_type = config['mpc']['optim_type']

    if planner_type == 'mppi':
        return PlannerMPPI(config)
