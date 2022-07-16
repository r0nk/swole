#!/usr/bin/python3 -u
import gym
import _thread
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

env = gym.make('CartPole-v1')

model = PPO('MlpPolicy',env,verbose=1)

should_run=True

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        print("training started.")

    def _on_step(self) -> bool:
        if(not should_run):
            print("stopping early...")
        return should_run

    def _on_training_end(self) -> None:
        print("training finished.")

callback=CustomCallback()

def learn_wrapper(steps,callback):
    model.learn(total_timesteps=steps,callback=callback)

while True:
    line = input('o> ')
    if len(line.split())==0:
        continue
    command = line.split()[0]
    if command == 'hello':
        print('world!')
        continue
    if command == 'learn':
        should_run=True
        timesteps=5000
        if(len(line.split())>1):
            timesteps=int(float(line.split()[1]))
        _thread.start_new_thread(learn_wrapper,(timesteps,callback,))
        continue
    if command == 'stop':
        should_run=False
        continue
    if command == 'save':
        save_file = 'model'
        if(len(line.split())>1):
            save_file=line.split()[1]
        model.save(save_file)
        continue
    if command == 'load':
        save_file = 'model'
        if(len(line.split())>1):
            save_file=line.split()[1]
        model = PPO.load(save_file)
        continue
    print('Command not found')
