import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np
from mamujoco_maddpg.maml_mamujoco.centralized_q import Centralized_q
import os


class MetaLearner:

    def __init__(self, args, sampler, gamma=0.95, outer_lr=1e-5, tau=1.0, device='cpu'):
        self.task_sampler = sampler
        self.gamma = gamma
        self.outer_lr = outer_lr
        self.tau = tau
        self.device = device
        self.centralized_q = Centralized_q(args=args, task_sampler=self.task_sampler).to(self.device)
        self.target_centralized_q = Centralized_q(args=args, task_sampler=self.task_sampler).to(self.device)
        path = "/Users/songrui/Machine Learning/Generalized_MADDPG/mamujoco_maddpg/maml_mamujoco/MAML_result/centralized_q_params_1.pth"
        if os.path.exists(path):
            self.centralized_q.load_state_dict(torch.load(path))
            self.target_centralized_q.load_state_dict(torch.load(path))
        self.centralized_q_optim = torch.optim.Adam(self.centralized_q.parameters(), lr=self.outer_lr)
        self.input_shape = self.centralized_q.input_shape
        # args.scenario_name = "simple_spread"
        # _, args = me.make_env(args=args)
        #
        # self.args = args
        self.train_step = 0
        self.total_training_step = 2000
        self.update_times = 1000
        self.episode_limit = 1000
        self.num_tasks = 4
        self.save_rate = 10
        self.load_rate = 1
        self.save_path = "./MAML_result"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        result = []
        inner_returns=[]
        inner_result = []
        c = 0
        for i in range(self.total_training_step):
            print("Meta Training " + str(i + 1) + " sampling " + str(self.num_tasks) + " tasks")
            tasks = self.task_sampler.sample()
            for time_step in range(self.update_times):
                c += 1
                total_q_loss = None
                for j, t in enumerate(tasks):
                    # inner training
                    # print("\tIn task " + str(j + 1) + " sampling " + str(self.task_sampler.batch_size) + " trajectory")
                    # load generalized centralized q function
                    for a in t.agents:
                        # for target_param, param in zip(a.policy.critic_target_network.parameters(),
                        #                                self.centralized_q.parameters()):
                        #     target_param.data.copy_(
                        #         (1 - self.args.tau) * target_param.data + self.args.tau * param.data)
                        if time_step == 0:
                            a.policy.critic_target_network.load_state_dict(self.centralized_q.state_dict())
                        if time_step % self.load_rate == 0:
                            a.policy.critic_network.load_state_dict(self.centralized_q.state_dict())
                    # inner training
                    task_q_loss = t.run(time_step=time_step, centralized_q=self.centralized_q)
                    if total_q_loss is None:
                        total_q_loss = task_q_loss
                    else:
                        total_q_loss = total_q_loss.add(task_q_loss)
                # if time_step > 0 and time_step % 250 == 0 and i % 100 == 0:
                #     inner_result.append([c, np.mean(inner_returns)])
                #     np.save("./MAML_result/inner_returns.npy", np.array(inner_result))
                if total_q_loss is not None:
                    self.centralized_q_optim.zero_grad()
                    total_q_loss.backward()
                    self.centralized_q_optim.step()


            returns = []
            for t in tasks:
                r = t.evaluate()
                returns.append(r)
            # min_return = min(returns)
            # if min_return<0:
            #     abs_min_return = abs(min_return)
            #     returns = [i+abs_min_return for i in returns]
            to_save = [i + 1, np.mean(returns)]
            result.append(to_save)

            if i % self.save_rate == 0:
                print("Saving training information and meta centralized q function parameters", end=" ")
                np.save("MAML_result/training_info_1.1.npy", np.array(result))
                torch.save(self.centralized_q.state_dict(), './MAML_result/centralized_q_params_1.pth')
                print("and successfully saved")

            print("Meta Update: " + str(i + 1), "\n\tinner_batch_avg_validation_return: " + str(np.mean(returns)))

    def test(self):
        self.centralized_q.load_state_dict(torch.load('./MAML_result/centralized_q_params.pth'))
        t = self.task_sampler.sample_test()
        for time_step in range(self.update_times):
            total_q_loss = None

            for a in t.agents:
                if time_step == 0:
                    a.policy.critic_target_network.load_state_dict(self.centralized_q.state_dict())
                a.policy.critic_network.load_state_dict(self.centralized_q.state_dict())
            # inner training
            task_q_loss = t.run(time_step=time_step, centralized_q=self.centralized_q)
            if total_q_loss is None:
                total_q_loss = task_q_loss
            else:
                total_q_loss = total_q_loss.add(task_q_loss)
            if total_q_loss is not None:
                self.centralized_q_optim.zero_grad()
                total_q_loss.backward()
                self.centralized_q_optim.step()
        r = t.evaluate()
        return r
