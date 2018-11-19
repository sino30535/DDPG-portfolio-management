# DDPG-stock-market-test
Build DDPG models and test on stock market

# Reference
* Code from original paper https://github.com/ZhengyaoJiang/PGPortfolio
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

# Results
* The models were built in a time series rolling scheme, using data from previous month to build rl model and test on next month.
* The model reach 14% rate of return from 2018/02/01 to 2018/10/29, compare with 5.6% rate of return using strategy of buy 15 stocks
  uniformly and hold and -16.8% rate of return using strategy of buy best performance stock in last month. More details in ipython           notebook.
* The rl model in stock market can be very unstable and suffered a lot from overfitting.
* The model only buy and sell with very small portion of portfolio, it didn't change position very often during the month.

