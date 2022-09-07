import base64, io
import numpy as np
import torch
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
from collections import namedtuple, deque
import glob
import os
import gym

class Train():
    """Train of DQL or DDQL"""
    
    def __init__(self, env, agent, n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.env = env
        self.agent = agent
        
    def train(self):
        scores = []# list containing scores from each episode
        avg_scores = []# list avg scores
        scores_window = deque(maxlen=100)# last 100 scores
        eps = self.eps_start # initialize epsilon
        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)# save most recent score
            avg_scores.append(np.mean(scores_window))# save current avg score
            scores.append(score)# save most recent score
            eps = max(self.eps_end, self.eps_decay*eps)# decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                if (self.agent.ddqn):
                    torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint_ddqn.pth')
                else:
                    torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint_dqn.pth')
                break
        return scores, avg_scores
    
    def show_video(self, env_name):
        """play video in MP4.

        Parameters
        ======
          env_name (int): name environment
        """
        mp4list = glob.glob('*.mp4')
        if len(mp4list) > 0:
            mp4 = '{}.mp4'.format(env_name)
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            display.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                  </video>'''.format(encoded.decode('ascii'))))
        else:
            print("Could not found video")

    def show_video_of_model(self, agent, env_name):
        """Salva video em MP4.

        Parâmetros
        ======
          agent (Agent): obj agent
          env_name (str): name ofenvoiroment
        """
        env = gym.make(env_name)
        vid = video_recorder.VideoRecorder(env, path="{}.mp4".format(env_name))
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint_dqn.pth'))
        state = env.reset()
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            vid.capture_frame()
            action = agent.act(state)
            state, reward, done, _ = env.step(action)        
        env.close()