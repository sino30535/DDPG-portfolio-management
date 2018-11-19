"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
from __future__ import print_function

from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
import gym.spaces


eps = 1e-20

dataset = pd.read_csv("data/stock_price_minutes.csv")
df_time = pd.DataFrame({'year': dataset.date.astype(str).str.slice(0, 4).astype(int),
                        'month': dataset.date.astype(str).str.slice(4, 6).astype(int),
                        'day': dataset.date.astype(str).str.slice(6, 8).astype(int),
                        'hour': dataset.time.astype(str).str.slice(0, 2).astype(int),
                        'minute': dataset.time.astype(str).str.slice(2, 4).astype(int)
                        })
df_time = pd.to_datetime(df_time).to_frame()
df_time.columns = ['Time']

def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, abbreviation, steps=120, window_length=50):
        """

        Args:
            history: (num_stocks, timestamp, 5) open, close, high, low, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 hours
            window_length: observation window
        """
        assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        import copy

        self.steps = steps + 1
        self.window_length = window_length

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)

    def _step(self):
        # get observation matrix from history
        # the stock data were observed every 7 minutes
        self.step += 7
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step + 7 >= self.steps
        return obs, done, ground_truth_obs

    def _step_sim(self):
        # get observation without stepping forward
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step + 7 >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0

        # get data for this episode, each episode might be different.
        self.idx = np.random.randint(
            low=self.window_length, high=self._data.shape[1] - self.steps)

        # print('Start date: {}'.format(index_to_date(self.idx)))
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 7, :]
        # apply augmentation?
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), returns_list=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.returns_list = returns_list
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0
        n_assests = len(self.asset_names)

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum() # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        LongPosition_value = p0 * (1 - mu1) * np.dot(y1[:n_assests+1], w0[:n_assests+1])
        ShortPosition_value = p0 * (1 - mu1) * np.dot(1-(y1[n_assests+1:]-1), w0[n_assests+1:])

        p1 = LongPosition_value + ShortPosition_value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns

        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return

        # TODO use sharpe ratio for reward
        reward = r1/self.steps  # reward
        # remember for next step
        self.p0 = p1
        self.w0 = w1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "weights": np.round(w1, 4),
            "cost": mu1,
            "long_position": LongPosition_value,
            "short_position": ShortPosition_value,
            "price_change": y1

        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names) * 2)
        self.p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 abbreviation,
                 steps=120,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0
                 ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2018-01-02' of the dataset
        """
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx

        self.src = DataGenerator(history, abbreviation, steps=steps, window_length=window_length)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names)*2 + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation), window_length,
                                                                                 history.shape[-1]), dtype=np.float32)
        
    def step(self, action, simulation=0):
        return self._step(action, simulation)

    def _step(self, action, simulation):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names)*2 + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), [1.0], 3, err_msg='weights should sum to 1. action="%s"' % weights)

        if simulation == 0:
            observation, done1, ground_truth_obs = self.src._step()
        else:
            observation, done1, ground_truth_obs = self.src._step_sim()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        # relative price vector of last observation day (close/open)
        current_open_price_vector = observation[:, -1, 0]
        last_open_price_vector = observation[:, -7, 0]

        y1 = current_open_price_vector / last_open_price_vector
        y1 = np.append(y1, y1[1:])

        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        if simulation == 0:
            info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
            # add dates
            info['date'] = self.start_idx + self.src.idx + self.src.step
            info['steps'] = self.src.step
            info['next_obs'] = ground_truth_obs
            self.infos.append(info)

        return observation, reward, done1 or done2, info
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info['next_obs'] = ground_truth_obs
        return observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'human':
            self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        df_env_info = pd.merge(df_info, df_time, left_on='date', right_index=True)
        x = range(len(df_env_info))
        tick_f = round(len(df_env_info)/15)
        plt.figure(1)
        plt.plot(x, df_env_info['portfolio_value'], label='Portfolio value')
        plt.plot(x, df_env_info['market_value'], label='Market value')
        plt.xticks(x[::tick_f], df_env_info.Time[::tick_f], rotation=30)
        plt.legend(loc='upper right')
        plt.title('Portfolio value')
        plt.show()


