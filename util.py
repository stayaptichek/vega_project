import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def predict_probs(model, states):
    with torch.no_grad():
        tens = torch.from_numpy(states).type(torch.float32)
        return F.softmax(model(tens)).cpu().numpy()

def generate_session_ES(env, agent, t_max=500):
    """
    Generates one session. 
    
    Args:
        env: gym environment
        agent: agent which interacts with environment
        t_max: trajectory length
        
    Returns:
        total_reward: total episode reward 
    """
    total_reward = 0
    state = env.reset()
    
    for t in range(t_max):
        action = agent.get_action(torch.tensor(state).float())
        new_state, reward, done, info = env.step(action)
        total_reward += reward
        state = new_state
        
        if done:
            break
            
    return total_reward


def generate_session(env, model, opt, t_max=1000):
    states, actions, rewards = [], [], []
    s = env.reset()

    for t in range(t_max):
        action_probs = predict_probs(model, np.array([s]))[0]

        a = np.random.choice(env.action_space.n, p=action_probs)
        new_s, r, done, info = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done:
            break

    return opt, env, model, states, actions, rewards

def get_cumulative_rewards(
    rewards,  # rewards at each step
    gamma=0.99  # discount for reward
):
    T = len(rewards)
    G = np.zeros(T)
    G[-1] = rewards[-1]
    for i in range(T - 2, -1, -1):
        G[i] = rewards[i] + gamma * G[i + 1]
    return G

def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot


def train_on_session(opt, env, model, states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)
    
    log_probs_for_actions = torch.sum(
        log_probs * to_one_hot(actions, env.action_space.n), dim=1)
   
    entropy = torch.sum(probs*log_probs)
    loss = -(torch.mean(log_probs_for_actions * cumulative_returns) + entropy_coef * entropy)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return np.sum(rewards)


def score(env, agent, n=10, t_max=500):
    """
    Calculates total score for given number of generated sessions n and 
    length of trajectory t_max. 
    
    Args:
        env: gym environment
        agent: agent which interacts with environment
        n: number of generated session 
        t_max: trajectory length
    
    Returns:
        total_reward: average for all episodes reward
    """
    
    rewards = [generate_session_ES(env, agent, t_max=t_max) for _ in range(n)]
    return sum(rewards) / n