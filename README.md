# DDPG-stock-market-test
Build DDPG models and test on stock market

# Reference
* Original paper https://github.com/ZhengyaoJiang/PGPortfolio
* https://github.com/vermouth1992/drl-portfolio-management
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059)
* The environment is inspired by https://github.com/wassname/rl-portfolio-management
* DDPG implementation is inspired by http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html

# Dataset
* 15 Stock price data from 2018/1/1 to 2018/10/29, recorded in minutes with open, close, high, low, volumn features, downloaded from https://www.finam.ru/profile/moex-akcii/gazprom/export/, BATS global markets.

# Basic settings
* The action contains cash position, 15 stock's long position, and 15 stock's short position.
* Observe stock price data every minutes, but only act in every 7 minutes.
* In each step, in addition to the original (s, a, r, s') , other state-action pairs of'inferred steps' were also collected and store in     the replay memory buffer.
