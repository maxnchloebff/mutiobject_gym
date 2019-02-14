import numpy as np

from gym.envs.robotics import rotations, multi_object_robot_env, utils
from xml.etree.ElementTree import ElementTree, Element


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class GraspEnv(multi_object_robot_env.MultiObjectEnv):
    """Superclass for all Grasp environments.
    """
    # some more parameters:
    # n_objects;shape_list;color_list;
    # modify some methods
    # _get_obs;
    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,n_objects,shape_list,color_list,
        desired_shape,desired_color
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            n_objects (int): the number of objects
            shape_list (string list): the shape of each object
            color_list (string list): the color of each object
            desired_shape (string): the shape we want to grasp
            desired_color (string): the color we want to grasp
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        # properties added
        self.n_objects = n_objects
        self.shape_list = shape_list
        self.color_list = color_list
        self.desired_shape = desired_shape
        self.desired_color = desired_color
        self._add_objects2xml(xml_path="/home/kai/gym/gym/envs/robotics/assets/grasp/different_shapes.xml",shape=shape_list,color=color_list,num_objects=n_objects)

        super(GraspEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)
        # print(self.get_sensor_sensordata())
    # GoalEnv methods
    # ----------------------------
    def _get_goal(self,obs,desired_shape):
        # this method get the positon where the desired shape is
        # return the center position of the shape
        position =  obs,desired_shape
        # get the pos from obs
        return np.zeros(3,dtype='float32')
    def compute_reward(self, achieved_goal, desired_goal, info):
        # run this method after get the achieved_goal by _get_goal()
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            #规定reward的类型，如果是sparse那么返回的是1和-1，如果是dense的话，就是连续值
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # MultiObjectEnv methods
    # ----------------------------
    def get_sensor_sensordata(self,body_name=None):
        return self.sim.data.sensordata

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        # 得到机器人的位置和速度非常简单，只需要调用位于utils中的robot_get_obs方法即可
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_list = [1]*self.n_objects
            for i in range(self.n_objects):
                object_pos = self.sim.data.get_site_xpos('object0')
                # rotations
                # Convert Rotation Matrix to Euler Angles
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                # velocities
                object_velp = self.sim.data.get_site_xvelp('object0') * dt
                object_velr = self.sim.data.get_site_xvelr('object0') * dt
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp
                object_list[i] = np.concatenate([object_pos, object_rot, object_velp, object_velr, object_rel_pos, object_velp])
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, gripper_state, grip_velp, gripper_vel, object_list[0],object_list[1],object_list[2]
        ])

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = self._get_goal(obs,self.desired_shape)


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        id = 'target'
        site_id = self.sim.model.site_name2id(id)
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        print(self.get_sensor_sensordata())
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            self.objects_qpos_ls =[1]*self.n_objects#每一个位置存放的是7位的数组
            for i in range(self.n_objects):
                object_xpos = self.initial_gripper_xpos[:2]
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    # 使得爪子和物体之间的距离大于0.1
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                #先从模型中得到xml文件中的object_qpos
                object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
                # 将object_qpos存在ｌｉｓｔ中
                self.objects_qpos_ls[i] = object_qpos.copy()
                # 检查是否是7长度
                assert object_qpos.shape == (7,)
                # 将前三位制成object_xpos
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
                obs = self._get_obs()
            self.desired_object_pos = self._get_goal(obs,self.desired_shape)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal_tem = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # add the 补偿
            goal_tem += self.target_offset
            goal_tem[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal_tem[2] += self.np_random.uniform(0, 0.45)
            goal = goal_tem
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        # 执行的第一个的方法，在seed之后
        # 将每一关节的信息从initial_qpos中赋值到sim中,其中就包括了object:joint的pos信息
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _add_objects2xml(self,xml_path,shape,color,num_objects):
        # find the root element mujoco, save it into doc
        doc = self.read_xml(xml_path)
        # find the list of element under and including worldbody element
        worldbody = self.find_nodes(doc,"worldbody")
        body = self.find_nodes(doc,"worldbody/body")
        element_worldbody = self.find_first_node_by_tagname(node_list=worldbody,tag="worldbody")
        index = 0
        # 删除所有原有的object
        while True:
            old_object = self.find_first_node_by_kv_map(node_list=body,tag="body",kv_map={"name":"object{}".format(index)})
            if old_object is not None:
                element_worldbody.remove(old_object)
                index += 1
            else:
                break;

        #新建特定个数量的物体
        for i in range(num_objects):
            new_body = self.create_node("body", {"name": "object{}".format(i), "pos": "0.025 0.025 0.025"})
            new_joint = self.create_node("joint",{"name": "object{}:joint".format(i), "type": "free", "damping": "0.01"})
            new_geom = self.create_node("geom", {"name": "object{}".format(i), "size": "0.02 0.02 0.02","type":self.shape_list[i],
                                                 "condim": "3", "material": "block_mat", "mass": "2"})
            new_site = self.create_node("site", {"name": "object{}".format(i), "pos": "0 0 0",
                                                 "size": "0.025 0.025 0.025", "rgba": self.color_list[i],"type":self.shape_list[i]})
            self.add_child_node(new_body, new_joint)
            self.add_child_node(new_body, new_geom)
            self.add_child_node(new_body, new_site)
            self.add_child_node(element_worldbody, new_body)
            self.write_xml(doc, xml_path)


        self.write_xml(doc, xml_path)
    # the following mothods process xml document
    def read_xml(self, in_path):
        '''''读取并解析xml文件
           in_path: xml路径
           return: ElementTree'''
        tree = ElementTree()
        tree.parse(in_path)
        return tree

    def write_xml(self,tree, out_path):
        '''''将xml文件写出
           tree: xml树
           out_path: 写出路径'''
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    def if_match(self,node, kv_map):
        '''''判断某个节点是否包含所有传入参数属性
           node: 节点
           kv_map: 属性及属性值组成的map'''
        for key in kv_map:
            if node.get(key) != kv_map.get(key):
                return False
        return True

    # ---------------search -----

    def find_nodes(self,tree, path):
        '''''查找某个路径匹配的所有节点
           tree: xml树
           path: 节点路径'''
        return tree.findall(path)

    def find_first_node_by_tagname(self,node_list,tag):
        for node in node_list:
            if node.tag == tag:
                return node

    def find_first_node_by_kv_map(self,node_list,tag,kv_map):
        for node in node_list:
            if node.tag == tag and self.if_match(node,kv_map):
                return node

    # ---------------change -----

    def create_node(self,tag, property_map):
        '''''新造一个节点
           tag:节点标签
           property_map:属性及属性值map
           content: 节点闭合标签里的文本内容
           return 新节点'''
        element = Element(tag, property_map)
        return element

    def add_child_node(self,fathernode, element):
        '''''给一个节点添加子节点
           fathernode: 父节点
           element: 子节点'''
        fathernode.append(element)