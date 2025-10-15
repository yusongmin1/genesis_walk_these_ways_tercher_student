from legged_gym import *
from warnings import WarningMessage

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import *
from legged_gym.utils.helpers import class_to_dict
from collections import deque
from scipy.stats import vonmises


class GO2_SpringJump(LeggedRobot):
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()
        self.command[self.episode_length_buf==self.command_frame,2]=1.0
        # compute observations, rewards, resets, ...
        self.check_jump()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]
        env_ids = self.not_pushed_up * ~self.has_jumped * (self.command[:,2]==1)
        self.max_height=torch.maximum(self.max_height,self.simulator.base_pos[:,2])
        # print(self.count,self.prob)
        if self.cfg.domain_rand.push_towards_goal and torch.any(env_ids):
            self._push_robots_upwards(env_ids)
            self.not_pushed_up[env_ids] = False        
        # env_ids =~self.not_pushed_up*(self.command[:,2]==1)*self.not_pushed_rotot
        # if self.cfg.domain_rand.push_towards_goal and torch.any(env_ids):
        #     self._push_robots_desired(env_ids)
        #     self.not_pushed_rotot[env_ids] =False
    def check_jump(self):
        """ Check if the robot has jumped
        """
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        # print("contact",contact)
        self.contact_filt =torch.logical_or(contact, self.last_contacts)
        self.last_contacts=contact.clone()

        jump_filter = torch.all(~self.contact_filt , dim=1)

        was_in_flight=torch.logical_and(jump_filter,self.command[:,2]>0)
        self.was_in_flight[was_in_flight] = True
        has_jumped = torch.logical_and(torch.any(self.contact_filt ,dim=1), self.was_in_flight) #飞起来过并且落地就是已经跳跃过了
        
        self.landing_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.simulator.base_pos[torch.logical_and(has_jumped,~self.has_jumped),:2]
        # Only count the first time flight is achieved:
        self.has_jumped[has_jumped] = True 

    def _push_robots_upwards(self,env_ids):

        random_push = torch.randint(0,10,(self.num_envs,1),device=self.device).squeeze()
        self.prob=max(8-int(self.count/(24*50)),0)
        env_ids = torch.logical_and(random_push < self.prob, env_ids)
        # 将布尔掩码转换为索引

        dofs_vel = self.simulator.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(1.5,2.2, (self.num_envs, 1), device=self.device).flatten()[env_ids]
        dofs_vel[env_ids, 2] += push_vel
        self.simulator.robot.set_dofs_velocity(dofs_vel)


    # def _push_robots_desired(self,env_ids):
    #     """ Randomly pushes some robots towards the goal just before takeoff. Emulates an impulse by setting a randomized base velocity. 
    #     """
        
    #     random_push = torch.randint(0,10,(self.num_envs,1),device=self.device).squeeze()
    #     env_ids = torch.logical_and(random_push<self.prob,env_ids)
    #     self.root_states[env_ids,7] += torch_rand_float(0.0,1.0, (self.num_envs, 1), device=self.device).flatten()[env_ids]
    #     # self.root_states[env_ids,11] -= 0.1
    #     self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))  


    def compute_observations(self):
        """ Computes observations
        """
        obs_buf = torch.cat((
            torch.zeros((self.num_envs, 2), device=self.device),
            self.command,    # cmd     3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,   # omega   3
            self.simulator.base_euler,                  # g       3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                       # p_t     12
            self.simulator.dof_vel * self.obs_scales.dof_vel,        # dp_t    12
            self.actions,                                  # a_{t-1} 12
        ), dim=-1) #3+3+3+12+12+12+4++1+1+1+4=

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:  # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((
                obs_buf,                                       # all above
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,   # v_t     3
                # domain randomization parameters
                self.simulator._rand_push_vels[:, :2],                   # 2
                self.simulator._added_base_mass,                         # 1
                self.simulator._friction_values,                         # 1
                self.simulator._base_com_bias,                           # 3
                # ctrl_delay,                                    # 1
                self.simulator._kp_scale,                                # 12
                self.simulator._kd_scale,                                # 12
                self.simulator._joint_armature,                          # 1
                self.simulator._joint_stiffness,                         # 1
                self.simulator._joint_damping,                           # 1
                # privileged infos
            ), dim=-1)
        # print(self.privileged_obs_buf.shape,obs_buf.shape)
        # add noise if needed
        if self.add_noise:
            obs_now = obs_buf.clone()
            obs_now += (2 * torch.rand_like(obs_now) - 1) * \
                self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()

        self.obs_history.append(obs_now)
        self.obs_buf = torch.cat(
            [self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=-1
        )
        self.critic_history.append(self.privileged_obs_buf)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.critic_history.maxlen)], dim=-1
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # self._resample_commands(env_ids)

        self.command[env_ids,:]=0.
        self._reset_dofs(env_ids)
        self.simulator.reset_idx(env_ids)
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.landing_poses[env_ids,:] = self.simulator.init_state[env_ids,:2]
        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # Periodic Reward Framework buffer reset

        self.last_contacts[env_ids] = 0
        self.not_pushed_up[env_ids] = True
        self.not_pushed_rotot[env_ids] =True
        self.command_frame=torch.randint(50,100,(self.num_envs,),device=self.device)
        self.max_height[env_ids]=0.0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        
        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        # clear obs and critic history for the envs that are reset
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:5] = 0.  # commands
        noise_vec[5:8] =noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel 
        noise_vec[8:11] = noise_scales.gravity * noise_level
        noise_vec[11:11+1*self.num_actions] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[11+1*self.num_actions:11+2*self.num_actions] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel  # dp_t
        noise_vec[11+2*self.num_actions:11+3*self.num_actions] = 0.  # a_{t-dt}

        return noise_vec

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        # obs_history
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.single_num_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )

######################reward#########################
    def _reward_before_setting(self):
        #切换到蹲姿状态之前的奖励函数
        rew = torch.exp(-torch.sum(torch.abs(self.simulator.dof_pos-self.simulator.default_dof_pos),dim=1)/2)*(self.command[:, 2] == 0)*(self.simulator.base_euler.sum(dim=1)<0.6)
        return rew
    
    def _reward_line_z(self):
        #在初始化后和落地之前z轴线速度越大越好
        rew=(self.simulator.base_lin_vel[:, 2]>0)*self.simulator.base_lin_vel[:, 2] *(~self.has_jumped)*(self.command[:, 2] == 1)
        return rew
    
    def _reward_land_pos(self):
        #xy轴角速度奖励
        land_err=self.simulator.init_state[:,:2]+self.command[:,:2]-self.landing_poses
        # print(land_err[0])
        return torch.exp(-torch.sum(torch.abs(land_err),dim=1))*(self.has_jumped)*(self.simulator.base_euler.sum(dim=1)<0.6)*(self.max_height>0.42)


    def _reward_base_height_flight(self):
        #跳跃的高度奖励
        base_height_flight = (self.simulator.base_pos[:, 2] - 0.5)
        rew= torch.exp(-torch.abs(base_height_flight)*5)*(self.was_in_flight)*~self.has_jumped*6
        return rew 
    
    def _reward_base_height_stance(self):
        #落地后的高度奖励和默认关节角度的奖励
        return torch.abs((self.simulator.base_pos[:, 2] - 0.3))*self.has_jumped + 0.2*torch.abs((self.simulator.base_pos[:, 2] - 0.25))*(self.command[:,2]==0)
    
    def _reward_dof_pos(self):
        #落地后的高度奖励和默认关节角度的奖励
        return torch.abs(self.simulator.dof_pos - self.simulator.default_dof_pos).sum(dim=1)

    def _reward_dof_hip_pos(self):
        #落地后的高度奖励和默认关节角度的奖励
        rew=torch.abs(self.simulator.dof_pos - self.simulator.default_dof_pos)
        return rew[:,0]+rew[:,3]+rew[:,6]+rew[:,9]
        
    def _reward_orientation(self):
        #rewer=self.base_euler_xyz
        rew=torch.exp(-torch.abs(self.simulator.base_euler).sum(dim=1))
        return rew

    def _reward_ang_vel_xy(self):
        rew=torch.sum(torch.abs(self.simulator.base_ang_vel),dim=1)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.abs(self.simulator.torques), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
    
    def _reward_collision(self):
        return torch.sum(1.*(torch.linalg.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_contact_forces(self):
        """
        Penalize foot contact when not expected
        """
        return (self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] -self.cfg.rewards.contact_force_threshold).clamp(min=0).mean(dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        self.count+=1
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_flight(self):
        # Penalize dof velocities
        return self.was_in_flight
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.command[:, 0] - self.simulator.base_lin_vel[:, 0])
        # print(self.commands[0, 0]*2 , self.base_lin_vel[0, 0])
        return torch.exp(-lin_vel_error)*self.was_in_flight*~self.has_jumped*5
    
    def _reward_line_vel_stance(self):
        # Penalize dof velocities
        return torch.sum(torch.abs(self.simulator.base_lin_vel[:,:2]), dim=1)*(self.has_jumped)
    

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.simulator.feet_pos - self.simulator.base_pos[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.simulator.feet_indices), 3, device=self.device)
        for i in range(len(self.simulator.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.simulator.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.abs(footpos_in_body_frame[:, :, 2] +0.20)
        return torch.sum(height_error, dim=1) *self.was_in_flight*(~self.has_jumped) *6
    
    def _reward_has_jump_contact(self):
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2]>1
        return (torch.sum(contact, dim=1)!=4)*self.has_jumped
