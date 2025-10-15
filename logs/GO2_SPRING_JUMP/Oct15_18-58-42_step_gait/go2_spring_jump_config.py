from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2_SpringJumpCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        frame_stack = 10   # policy frame stack
        c_frame_stack = 3  # critic frame stack
        num_single_obs = 47
        num_observations = int(num_single_obs * frame_stack)
        single_num_privileged_obs = 84
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        env_spacing = 1.0
        episode_length_s = 5 # episode length in seconds

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RR_hip_joint': 0.,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25  # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4  # decimation: Number of control action updates @ sim DT per policy DT

    class asset(LeggedRobotCfg.asset):
        # Common
        name = "go2" # name of the robot
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        # For Genesis
        dof_names = [           # align with the real robot
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint"
        ]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        only_positive_rewards=False
        reward_sigma=0.25
        contact_force_threshold = 100
        class scales(LeggedRobotCfg.rewards.scales):
            # limitation
            before_setting=2.
            line_z=5.
            flight=1.
            base_height_flight=2.5
            base_height_stance=-10
            orientation=1.5
            dof_pos=-0.1
            dof_hip_pos=-0.2
            ang_vel_xy=-0.05
            torques=-0.0002
            dof_pos_limits=-10.
            dof_vel=-0.001
            termination=0.0
            collision=-1.
            action_rate=-0.01
            feet_contact_forces=-0.1
            land_pos=8
            tracking_lin_vel=0.5
            line_vel_stance=-0.5
            foot_clearance=-1
            has_jump_contact=-1.
            
            
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1, 1.]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        enable = True
        randomize_friction = enable
        friction_range = [0.2, 1.2]
        randomize_base_mass = enable
        added_mass_range = [-1., 2]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 1.0
        randomize_com_displacement = enable
        com_displacement_range = [-0.03, 0.03]
        randomize_pd_gain = enable
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = enable
        joint_armature_range = [0.0, 0.02]  # [N*m*s/rad]
        randomize_joint_stiffness = enable
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = enable
        joint_damping_range = [0.1, 0.3]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        push_towards_goal = True
    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.03
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

class GO2_SpringJumpCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        sym_loss = True
        obs_permutation = [-0.0001, -1, 2, -3, 4,
                           -5,6,-7,-8,9,-10,
                       -14,15,16,-11,12,13,-20,21,22,-17,18,19,
                       -26,27,28,-23,24,25,-32,33,34,-29,30,31,
                       -38,39,40,-35,36,37,-44,45,46,-41,42,43
                       ]
        ##command x y height
        act_permutation = [ -3, 4, 5, -0.0001, 1, 2, -9, 10, 11,-6, 7, 8,]#关节电机的对陈关系
        frame_stack = 10
        sym_coef = 1.0
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'step_gait'
        experiment_name = 'GO2_SPRING_JUMP'
        save_interval = 100
        load_run = "Oct15_07-39-24_step_gait"
        checkpoint = -1
        max_iterations = 20000
