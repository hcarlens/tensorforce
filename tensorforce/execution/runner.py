# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.execution.base_runner import BaseRunner

import time
from six.moves import xrange
import warnings
from inspect import getargspec
from tqdm import tqdm
from tensorboard_logger import log_value
import numpy as np 
import pommerman
from pommerman.agents import SimpleAgent

class Runner(BaseRunner):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0, use_simple_rewards=False):
        """
        Initialize a single Runner object (one Agent/one Environment).

        Args:
            id_ (int): The ID of this Runner (for distributed TF runs).
        """
        super(Runner, self).__init__(agent, environment, repeat_actions, history)
        self.simple_agent = SimpleAgent()
        self.id = id_  # the worker's ID in a distributed run (default=0)
        self.current_timestep = None  # the time step in the current episode
        self.use_simple_rewards = use_simple_rewards

    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False,
            episode_finished=None, summary_report=None, summary_interval=None, timesteps=None, episodes=None, testing=False, sleep=None
            ):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """

        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps
            warnings.warn("WARNING: `timesteps` parameter is deprecated, use `num_timesteps` instead.",
                          category=DeprecationWarning)
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)

        # figure out whether we are using the deprecated way of "episode_finished" reporting
        old_episode_finished = False
        if episode_finished is not None and len(getargspec(episode_finished).args) == 1:
            old_episode_finished = True

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()

        if num_episodes is not None:
            num_episodes += self.agent.episode

        if num_timesteps is not None:
            num_timesteps += self.agent.timestep

        # Update global counters.
        self.global_episode = self.agent.episode  # global value (across all agents)
        self.global_timestep = self.agent.timestep  # global value (across all agents)

        # add progress bar
        with tqdm(total=num_episodes) as pbar:
            # episode loop
            while True:
                episode_start_time = time.time()
                state = self.environment.reset()
                self.agent.reset()


                episode_reward = 0
                self.current_timestep = 0

                # initialise action counters
                # hardcoded action space size, soz
                # action space dimension wasn't available here
                action_counter = [0] * 6
                action_labels = ['pass','up','down','left','right','bomb']

                # did we win or not?
                episode_outcome = 0

                # time step (within episode) loop
                while True:
                    action = self.agent.act(states=state, deterministic=deterministic)
                    # increment action counter
                    action_counter[action] += 1

                    reward = 0
                    for _ in xrange(self.repeat_actions):
                        if self.use_simple_rewards:
                            #if simple agent took same move and simple agent wasn't random 
                            self.active_agent = self.environment.gym.training_agent
                            self.observation = (self.environment.gym.observations[self.active_agent]).copy()
                            
                        state, terminal, step_reward = self.environment.execute(action=action)
                        reward += step_reward
                        if terminal:
                            # if the terminal reward is possible, assume we won
                            if reward > 0:
                                episode_outcome = 1
                            break

                        if self.use_simple_rewards and (self.environment.gym.observations[self.active_agent]).get("position") is not None:
                            simple_action = self.simple_agent.act(self.observation, action_counter)
                            if simple_action == action and self.simple_agent.was_random == False:
                                reward += 1

                    # If an agent has died then give positive reward to alive agents


                    if max_episode_timesteps is not None and self.current_timestep >= max_episode_timesteps:
                        terminal = True

                    if not testing:
                        self.agent.observe(terminal=terminal, reward=reward)

                    self.global_timestep += 1
                    self.current_timestep += 1
                    episode_reward += reward

                    if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                        break

                    if sleep is not None:
                        time.sleep(sleep)

                # Update our episode stats.
                time_passed = time.time() - episode_start_time
                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.current_timestep)
                self.episode_times.append(time_passed)
                self.episode_outcomes.append(episode_outcome)

                ep_num = np.asscalar(self.global_episode)
                log_value('reward',self.episode_rewards[-1],ep_num)
                log_value('win', episode_outcome,ep_num)
                log_value('cumulative_reward',sum(self.episode_rewards),ep_num)
                log_value('cumulative_wins', sum(self.episode_outcomes),ep_num)
                log_value('100_ep_avg_win', np.mean(self.episode_outcomes[-100:]),ep_num)
                log_value('ep_timesteps',self.episode_timesteps[-1],ep_num)
                log_value('ep_times',self.episode_times[-1],ep_num)
                for i in range(0,len(action_counter)):
                    action_prevalence = action_counter[i] / self.episode_timesteps[-1]
                    log_value('action_freq_' + action_labels[i], action_prevalence, ep_num)

                self.global_episode += 1
                pbar.update(1)

                # Check, whether we should stop this run.
                if episode_finished is not None:
                    # deprecated way (passing in only runner object):
                    if old_episode_finished:
                        if not episode_finished(self):
                            break
                    # new unified way (passing in BaseRunner AND some worker ID):
                    elif not episode_finished(self, self.id):
                        break
                if (num_episodes is not None and self.global_episode >= num_episodes) or \
                        (num_timesteps is not None and self.global_timestep >= num_timesteps) or \
                        self.agent.should_stop():
                    break
            pbar.update(num_episodes - self.global_episode)

    # keep backwards compatibility
    @property
    def episode_timestep(self):
        return self.current_timestep


# more descriptive alias for Runner class
DistributedTFRunner = Runner
SingleRunner = Runner
