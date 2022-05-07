import numpy as np
from util import train_on_session, generate_session, score
import torch 
import torch.nn as nn
import time 
from tqdm import trange
from IPython.display import clear_output
from copy import deepcopy

class MLPPolicy(nn.Module):
    """
    Agent as a NN. Maps a state to an action. 
    """
    def __init__(self, n_features, n_actions, n_hiddens, discrete = False):
        super().__init__()
        self.discrete = discrete
        
        layers = []
        # use two linear layers
        layers.append(nn.Linear(n_features, n_hiddens, bias=False))        
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hiddens, n_hiddens, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hiddens, n_actions, bias=False))
        
        self.model = nn.Sequential(*layers)

        for param in self.parameters():
            param.data *= 0
            param.requires_grad = False
        
    def get_action(self, state):
        """
        Gets an action from the state. Takes argmax (most likely) action. 
        """
        with torch.no_grad():
            model_output = self.model(state)

        if self.discrete:
            return np.argmax(model_output.detach().numpy())
        
        return model_output.numpy()

class Reinforce:
    def __init__(self, agent, env, seed):
        self.env = env
        self.agent = agent
        self.opt = torch.optim.Adam(self.agent.parameters(), 1e-3)
        self.seed = seed
        
        self.log = []
        
    def step(self, i):
        rewards = [train_on_session(*generate_session(self.env, self.agent, self.opt)) for _ in range(100)]
        self.log.append((np.mean(rewards), i))
        
    def train(self, bound):
        torch.manual_seed(self.seed)
        for i in range(100):
            self.step(i)
            #print("Reinforce = {}".format(self.log[-1][0]))
            #clear_output(True)
            if self.log[-1][0] > bound:
                break
        
        
def add_noise_to_model(model, noise, copy=False):
    if copy:
        new_model = deepcopy(model)
    else:
        new_model = model
        
    for param, noise_param in zip(new_model.parameters(), noise):
        param.data += noise_param

    return new_model

class ES:
    def __init__(self, get_env_function, model, lr=0.001, std=0.01, n_samples = 64, n_threads = 1, normalize=True):
        super().__init__()
        self.lr = lr
        self.std = std
        self.normalize = normalize
        self.n_samples = n_samples
        self.mean_reward_history = []
        self.n_threads = n_threads
        self.env = get_env_function()
        self.model = model
        
        self.log = []
        
        
    def get_noised_model(self, model):
        """
        Generate noise and adds it to the model.
        
        Args:
            model: maps the sates into the actions (i.e. agent)
        
        Returns:
            model with noise and noise itself.
        """
        noise = []
        for param in model.parameters():
            noise.append(self.std * torch.randn(param.shape))
            
        return add_noise_to_model(model, noise, copy=True), noise
    
    def normalize_rewards(self, rewards):
        """
        Normalizes rewards. 
        
        Args:
            rewards: model rewards               
        """
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)
        return rewards

    
    def optimize(self, model, noises, rewards):
        """
        Updates weights by adding the combined (weighted) noise. 
        
        Args:
            model: maps the sates into the actions (i.e. agent)
            noises: list of vectors of noises 
            rewards: rewards                
        """
        if self.normalize:
            rewards = self.normalize_rewards(rewards)
        
        combined_noise = []
        for i in range(len(list(model.parameters()))):
            cat_noise = torch.cat([n[i].unsqueeze(n[i].dim()) for n in noises], dim=-1)
            combined_noise.append(torch.sum(rewards * cat_noise / self.std, dim=-1))
            combined_noise[-1] *= self.lr / (len(noises) * self.std)

        add_noise_to_model(model, combined_noise)
    
    def step(self, model, i):
        """
        Calculates rewards and makes an optimizing step.
        
        Args:
            model: maps the sates into the actions (i.e. agent)
        """
        st = time.time()
        rewards = []
        noises = []
        noised_models = []
        for i in range(self.n_samples):
            noised_model, noise = self.get_noised_model(model)
            noised_models.append(noised_model)
            noises.append(noise)
        
        if self.n_threads == 1:
            rewards = [score(self.env, noised_models[i]) for i in range(self.n_samples)]
        else:
            rewards = np.array(Parallel(n_jobs=self.n_threads)(delayed(score)(deepcopy(self.env), noised_models[i]) 
                                                               for i in range(self.n_samples)))
        
        self.optimize(model, noises, torch.tensor(rewards))
        
        self.log.append((np.mean(rewards), i))
        #self.update_log(rewards)
        
    def train(self, bound):
        for i in range(100):
            self.step(self.model, i)
            
            #print("ES = {}".format(self.log[-1][0]))
            #clear_output(True)
            
            if self.log[-1][0] > bound:
                break
    