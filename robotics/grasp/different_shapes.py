import os
from gym import utils
from gym.envs.robotics import grasp_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('grasp', 'different_shapes.xml')


class DifferentShapeEnv(grasp_env.GraspEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.]
        }
        grasp_env.GraspEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,n_objects=5,color_list=["1 1 0 1","0 1 1 1","1 0 1 1","1 0 1 1","0 1 1 1"],
            shape_list=["sphere","capsule","sphere","sphere","capsule"],
            desired_color='1 0 0 1',desired_shape='sphere')
        utils.EzPickle.__init__(self)