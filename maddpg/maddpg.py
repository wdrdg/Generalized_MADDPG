import torch
import os
from maddpg.actor_critic import Actor, Critic
import copy


class MADDPG:
    def __init__(self, args, agent_id, input_shape=0):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.input_shape = input_shape
        self.device = args.device
        # create the network

        self.actor_network = Actor(args, agent_id).to(self.device)
        self.critic_network = Critic(args, self.input_shape).to(self.device)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(self.device)
        self.critic_target_network = Critic(args, self.input_shape).to(self.device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        #record loss
        self.q_loss = None
        self.a_loss = None

        # create the dict for store the model_maml
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model_maml
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型

        # path of meta model_maml
        if self.args.load_meta == 1:
            path = "E:\Project\Generalized_MADDPG\maml_rl\MAML_result\centralized_q_params.pth"
            self.critic_network.load_state_dict(torch.load(path))
            self.critic_target_network.load_state_dict(torch.load(path))

        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        o_copy = copy.deepcopy(o)
        u_copy = copy.deepcopy(u)
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()

        # record the loss
        self.a_loss = actor_loss
        self.q_loss = critic_loss

        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1
        return o_copy, u_copy

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


