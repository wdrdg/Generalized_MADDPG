import numpy as np
import torch
from .maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args, input_shape=112):
        super(Agent, self).__init__()
        self.args = args
        self.device = args.device
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id, input_shape)


    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.device)
            pi = self.policy.actor_network(inputs).squeeze(0).to(self.device)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        return self.policy.train(transitions, other_agents)

