import json

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import seaborn as sns

from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import norm
from pykalman import KalmanFilter
from prettytable import PrettyTable
from itertools import combinations


class DataFetcher:
    def __init__(self, start_date, end_date):
        self.tickers = self.load_tickers()
        self.start_date = start_date
        self.end_date = end_date
    
    @staticmethod
    def load_tickers():
        with open("Tickers.json", "r") as json_file:
            tickers_json = json.load(json_file)
        return tickers_json["tickers"]

    def get_data(self, column='Close'):
        """
        Fetch historical stock data for a given ticker from Yahoo Finance
        """
        data = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date, progress=False)
        data = data[column].dropna(how='any')
        return data
    
    @staticmethod
    def split_data(price_data, volume_data, split_ratio=0.70):
        """
        Splits the data into training and testing sets based on the given split ratio
        """
        total_rows = len(price_data)
        train_rows = int(split_ratio * total_rows)
        train_data = price_data.iloc[:train_rows]
        test_data = price_data.iloc[train_rows:]
        volume_train_data = volume_data.iloc[:train_rows]
        volume_test_data = volume_data.iloc[train_rows:]
        return train_data, test_data, volume_train_data, volume_test_data


class CointegrationAnalysis:
    @staticmethod
    def cointegration(data):
        """
        Perform a cointegration test on all pairs of columns in the given DataFrame
        """
        n = data.shape[1]
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i, j in combinations(range(n), 2):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
        return pvalue_matrix, pairs


class PairProcessor:
    def __init__(self, lookback_period=200, step_size=20, periods_per_year=252):
        self.lookback_period = lookback_period
        self.step_size = step_size
        self.periods_per_year = periods_per_year
        self.data = None
        self.volume_data = None
        self.pvalues_df = None
        self.pairs = None
        self.tbill = None
        self.portfolio_data = {}
        self.sharpe_values_dict = {}

    def set_data(self, data, volume_data, pvalues_df, pairs, tbill):
        """
        Set or update the data needed for processing pairs
        """
        self.data = data
        self.volume_data = volume_data
        self.pvalues_df = pd.DataFrame(pvalues_df, index=data.columns, columns=data.columns)
        self.pairs = pairs
        self.tbill = tbill

    def process_all_pairs(self):
        """
        Process all pairs based on the provided pairs and p values
        """
        stacked_df = self.pvalues_df.stack()
        stacked_df.index.names = ['Asset1', 'Asset2']
        stacked_df = stacked_df.reset_index()
        stacked_df.columns = ['Asset1', 'Asset2', 'P-value']
        stacked_df = stacked_df.dropna(subset=['P-value'])
        stacked_df = stacked_df.reset_index(drop=True)
        sequential_results = [
            result for asset1, asset2 in self.pairs 
            for result in self.process_pair(asset1, asset2, self.data)
        ]
        results_df = pd.DataFrame(sequential_results, columns=['Asset1', 'Asset2', 'Start', 'End', 'Beta (Kalman)', 'Half-life'])
        simulator = PairsTradingSimulator()
        results, portfolio_data = simulator.simulate_pairs_trading(
            data=self.data, liquidity_data=self.volume_data, cointegration_results=results_df
        )
        self.portfolio_data = portfolio_data
        unique_results = results.drop_duplicates(subset=['Asset1', 'Asset2'])[['Asset1', 'Asset2', 'Terminal Wealth']]
        sorted_unique_results = unique_results.sort_values(by='Terminal Wealth', ascending=False)
        return sorted_unique_results
    
    @staticmethod
    def process_pair(asset1, asset2, data, lookback_period=200, step_size=20):
        """
        Analyze and process a pair of assets for cointegration and calculates the Kalman-filtered beta values and half-life of mean reversion
        """
        num_windows = (len(data) - lookback_period) // step_size + 1
        beta_kalman_values = np.empty(num_windows)
        half_life_values = np.empty(num_windows)
        X = np.array([data[asset1].iloc[start:start+lookback_period].values
                      for start in range(0, len(data) - lookback_period + 1, step_size)])
        Y = np.array([data[asset2].iloc[start:start+lookback_period].values
                      for start in range(0, len(data) - lookback_period + 1, step_size)])
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        for idx, (x, y) in enumerate(zip(X, Y)):
            obs_mat = np.expand_dims(np.vstack([[x], [np.ones_like(x)]]).T, axis=1)
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=trans_cov)
            state_means, _ = kf.filter(y[:, np.newaxis])
            beta_kalman_values[idx] = state_means[-1, 0]
            spread = y - beta_kalman_values[idx] * x
            lagged_spread = spread[:-1]
            delta_spread = spread[1:] - spread[:-1]
            spread_model = sm.OLS(delta_spread, lagged_spread).fit()
            half_life_values[idx] = -np.log(2) / spread_model.params[0]
        p_values = [adfuller(y - beta * x)[1] for y, x, beta in zip(Y, X, beta_kalman_values)]
        results = [(asset1, asset2, start, start + lookback_period, beta, hl)
                   for start, beta, hl, p in zip(range(0, len(data) - lookback_period + 1, step_size),
                                                beta_kalman_values, half_life_values, p_values) if p < 0.05]
        return results

    @staticmethod
    def test_pair(asset1, asset2, data, initial_beta, lookback_period=200, step_size=20):
        """
        Test a pair of assets for potential trading opportunities using Kalman Filter and cointegration tests
        """
        num_windows = (len(data) - lookback_period) // step_size + 1
        beta_kalman_values = np.empty(num_windows)
        half_life_values = np.empty(num_windows)
        X = np.array([data[asset1].iloc[start:start+lookback_period].values
                      for start in range(0, len(data) - lookback_period + 1, step_size)])
        Y = np.array([data[asset2].iloc[start:start+lookback_period].values
                      for start in range(0, len(data) - lookback_period + 1, step_size)])
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        for idx, (x, y) in enumerate(zip(X, Y)):
            obs_mat = np.expand_dims(np.vstack([[x], [np.ones_like(x)]]).T, axis=1)
            initial_state_mean = [initial_beta, 0] if idx == 0 else np.zeros(2)
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=initial_state_mean,
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=trans_cov)
            state_means, _ = kf.filter(y[:, np.newaxis])
            beta_kalman_values[idx] = state_means[-1, 0]
            spread = y - beta_kalman_values[idx] * x
            lagged_spread = spread[:-1]
            delta_spread = spread[1:] - spread[:-1]
            spread_model = sm.OLS(delta_spread, lagged_spread).fit()
            half_life_values[idx] = -np.log(2) / spread_model.params[0]
        p_values = [adfuller(y - beta * x)[1] for y, x, beta in zip(Y, X, beta_kalman_values)]
        results = [(asset1, asset2, start, start + lookback_period, beta, hl)
                   for start, beta, hl, p in zip(range(0, len(data) - lookback_period + 1, step_size),
                                                beta_kalman_values, half_life_values, p_values) if p < 0.05]
        return results
    
    def calculate_sharpe_ratios(self):
        """
        Calculate the Sharpe ratios for each asset pair in the portfolio data
        """
        for (asset1, asset2), portfolio_data in self.portfolio_data.items():
            total_asset_series = pd.Series(portfolio_data['total_asset'])
            daily_returns = total_asset_series.pct_change().dropna()
            computed_sharpe = FinancialMetrics.sharpe_ratio(daily_returns, self.tbill, self.periods_per_year)
            self.sharpe_values_dict[(asset1, asset2)] = computed_sharpe
        sharpe_values_df = pd.DataFrame(list(self.sharpe_values_dict.items()), columns=['Pair', 'Sharpe Ratio'])
        return sharpe_values_df

    def get_top_sharpe_pairs(self, top_n=5):
        """
        Retrieve the top N asset pairs based on Sharpe Ratio and print them
        """
        sharpe_values_df = self.calculate_sharpe_ratios()
        top_sharpe_values = sharpe_values_df.sort_values(by='Sharpe Ratio', ascending=False).head(top_n)
        top_pairs = top_sharpe_values['Pair'].tolist()
        filtered_portfolio_data = {pair: self.portfolio_data[pair] for pair in top_pairs}
        top_pair = list(filtered_portfolio_data.keys())[0]
        best_pair_data = filtered_portfolio_data[top_pair]
        print("Top Sharpe Ratios:")
        print(top_sharpe_values)
        return filtered_portfolio_data, top_pair, best_pair_data


class PositionManager:
    @staticmethod
    def zscore_tier_multiplier_upper(zscores):
        """
        Determine the position size multiplier based on the absolute z-score tier for an array of zscores
        """
        multipliers = np.zeros(zscores.shape)
        multipliers[(1 < zscores) & (zscores < 1.5)] = 0.5
        multipliers[(1.5 <= zscores) & (zscores < 2)] = 0.75
        multipliers[(2 <= zscores) & (zscores < 2.5)] = 0.9
        multipliers[zscores >= 2.5] = 1
        return multipliers

    @staticmethod
    def zscore_tier_multiplier_full(zscores):
        """
        Determine the position size multiplier based on the absolute z-score tier for an array of zscores
        """
        abs_zscores = np.abs(zscores)
        multipliers = np.zeros_like(abs_zscores)
        multipliers[(1 < abs_zscores) & (abs_zscores < 1.5)] = 0.5
        multipliers[(1.5 <= abs_zscores) & (abs_zscores < 2)] = 0.75
        multipliers[(2 <= abs_zscores) & (abs_zscores < 2.5)] = 0.9
        multipliers[abs_zscores >= 2.5] = 1
        return multipliers

    @staticmethod
    def adjusted_positions_vectorized(z_values, initial_capital, asset_prices, full):
        """
        Calculate the number of shares (including fractional shares) to buy for positions based on z-score magnitude
        """
        if full:
            multipliers = PositionManager.zscore_tier_multiplier_full(z_values)
        else:
            multipliers = PositionManager.zscore_tier_multiplier_upper(z_values)
        return (initial_capital * multipliers) / asset_prices


class PairsTradingSimulator:
    def __init__(self, transaction_cost_percent=0.1, stop_loss_percent_trade=0.95, take_profit_percent_trade=1.10, initial_capital=100, full=False):
        self.transaction_cost_percent = transaction_cost_percent
        self.stop_loss_percent_trade = stop_loss_percent_trade
        self.take_profit_percent_trade = take_profit_percent_trade
        self.initial_capital = initial_capital
        self.full = full

    def simulate_pairs_trading(self, data, liquidity_data, cointegration_results):
        """
        Simulates a pairs trading strategy on historical data, considering transaction costs, stop loss, and take profit parameters
        """
        portfolio_data = {}
        results_df = cointegration_results.copy()
        unique_pairs = results_df.drop_duplicates(subset=['Asset1', 'Asset2'])
        for index, row in unique_pairs.iterrows():
            asset1_lp = row['Asset1']
            asset2_lp = row['Asset2']
            capital_lp = self.initial_capital
            hedge_ratios = row['Beta (Kalman)']
            spread = data[asset2_lp] - hedge_ratios * data[asset1_lp]
            hl = max(1, int(row['Half-life']))
            mean_spread = spread.rolling(window=hl).mean()
            std_spread = spread.rolling(window=hl).std()
            z_values = (spread - mean_spread) / std_spread
            upper_limit = 1
            lower_limit = -1
            signals1 = np.where(z_values > 1, -1, np.where(z_values < -1, 1, 0))
            positions1 = np.diff(signals1, prepend=0)
            signals2 = -signals1
            positions2 = np.diff(signals2, prepend=0)
            holdings1 = np.zeros(len(data))
            cash1 = np.zeros(len(data))
            cash1[0] = capital_lp / 2
            holdings2 = np.zeros(len(data))
            cash2 = np.zeros(len(data))
            cash2[0] = capital_lp / 2
            slippage_series = []
            for t in range(1, len(data)):
                current_capital1 = cash1[t-1] + holdings1[t-1]
                current_capital2 = cash2[t-1] + holdings2[t-1]
                asset1_price = data[asset1_lp].iloc[t]
                asset2_price = data[asset2_lp].iloc[t]
                z_value = z_values.iloc[t] if isinstance(z_values, pd.Series) else z_values[t]
                adjusted_positions1 = PositionManager.adjusted_positions_vectorized(
                    z_value, current_capital1, asset1_price, self.full
                )
                adjusted_positions2 = PositionManager.adjusted_positions_vectorized(
                    z_value, current_capital2, asset2_price, self.full
                )
                holdings1[t] = holdings1[t-1] + positions1[t] * asset1_price * adjusted_positions1
                cash1[t] = cash1[t-1] - positions1[t] * asset1_price * adjusted_positions1
                holdings2[t] = holdings2[t-1] + positions2[t] * asset2_price * adjusted_positions2
                cash2[t] = cash2[t-1] - positions2[t] * asset2_price * adjusted_positions2
                transaction_cost1 = np.abs(positions1[t]) * self.transaction_cost_percent * data[asset1_lp].iloc[t]
                transaction_cost2 = np.abs(positions2[t]) * self.transaction_cost_percent * data[asset2_lp].iloc[t]
                liquidity_adjustment1 = self.calculate_liquidity_adjustment(liquidity_data, asset1_lp, t)
                liquidity_adjustment2 = self.calculate_liquidity_adjustment(liquidity_data, asset2_lp, t)
                transaction_cost1 *= liquidity_adjustment1
                transaction_cost2 *= liquidity_adjustment2
                slippage1 = self.calculate_slippage(liquidity_data[asset1_lp], positions1[t], t)
                slippage2 = self.calculate_slippage(liquidity_data[asset2_lp], positions2[t], t)
                slippage_tot = slippage1 + slippage2
                slippage_series.append(slippage_tot)
                cash1[t] -= (transaction_cost1 + slippage1)
                cash2[t] -= (transaction_cost2 + slippage2)
            total_asset1 = holdings1 + cash1
            total_asset2 = holdings2 + cash2
            total_asset = total_asset1 + total_asset2
            stop_loss = (total_asset < capital_lp * self.stop_loss_percent_trade).cumsum() > 0
            take_profit = (total_asset > capital_lp * self.take_profit_percent_trade).cumsum() > 0
            exit_trade = stop_loss | take_profit
            if exit_trade.any():
                holdings1[exit_trade] = 0
                holdings2[exit_trade] = 0
                total_asset = holdings1 + cash1 + holdings2 + cash2
            final_portfolio_lp = total_asset[-1]
            results_df.at[index, 'Terminal Wealth'] = round(final_portfolio_lp, 2)
            portfolio_data[(asset1_lp, asset2_lp)] = {
                'signals': {
                    'signals1': signals1,
                    'positions1': positions1,
                    'signals2': signals2,
                    'positions2': positions2,
                    'adjusted_positions1': adjusted_positions1,
                    'adjusted_positions2': adjusted_positions2
                },
                'hedge_ratios': hedge_ratios,
                'spread': spread,
                'z_values': z_values,
                'mean_spread': mean_spread,
                'std_spread': std_spread,
                'holdings1': holdings1,
                'cash1': cash1,
                'total_asset1': total_asset1,
                'holdings2': holdings2,
                'cash2': cash2,
                'total_asset2': total_asset2,
                'total_asset': total_asset,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit,
                'slippage': slippage_series
            }
        return results_df, portfolio_data

    def calculate_liquidity_adjustment(self, liquidity_data, asset, time_index):
        """
        Adjusts transaction cost based on liquidity for a specific asset
        """
        low_liquidity_threshold = liquidity_data[asset].quantile(0.25)
        high_liquidity_threshold = liquidity_data[asset].quantile(0.75)
        average_daily_volume = liquidity_data[asset].iloc[time_index]
        if average_daily_volume < low_liquidity_threshold:
            adjustment_factor = 1.5
        elif average_daily_volume > high_liquidity_threshold:
            adjustment_factor = 0.75
        else:
            adjustment_factor = 1.0
        return adjustment_factor

    def calculate_slippage(self, liquidity_data, position_size, time_index):
        """
        Calculate slippage based on the size of the position relative to the asset's liquidity
        """
        baseline_slippage_rate = 0.02
        liquidity_at_time = liquidity_data.iloc[time_index]
        proportion_of_trade = abs(position_size) / liquidity_at_time if liquidity_at_time != 0 else 0
        slippage_rate = baseline_slippage_rate * proportion_of_trade
        slippage_amount = position_size * slippage_rate
        return slippage_amount


class Plotter:
    @staticmethod
    def plot_cointegration_matrix(train_pvalues, train_data):
        """
        Plots the cointegration matrix of p-values between pairs.
        """
        mask = train_pvalues >= 0.05
        fig, ax = plt.subplots(figsize=(10,7))
        sns.heatmap(train_pvalues, xticklabels=train_data.columns, yticklabels=train_data.columns, 
                    cmap='Blues_r', annot=True, fmt=".2f", mask=mask, annot_kws={"size": 8})
        ax.set_title('Assets Cointegration Matrix p-values Between Pairs')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(correlation_matrix):
        """
        Plots the correlation matrix for cointegrated pairs.
        """
        fig, ax = plt.subplots(figsize=(10,7))
        sns.heatmap(correlation_matrix, ax=ax, cmap='Blues', annot=True, fmt=".2f", annot_kws={"size": 8})
        ax.set_title('Assets Correlation Matrix for Cointegrated Pairs')
        plt.show()

    @staticmethod
    def plot_pairs_trading_performance(portfolio_data, date_index):
        """
        Plots the trading performance of pairs trading strategies
        """
        for pair, portfolio in portfolio_data.items():
            total_asset_series = pd.Series(portfolio['total_asset'], index=date_index)
            z_values_series = pd.Series(portfolio['z_values'], index=date_index)
            upper_limit_series = pd.Series(portfolio['upper_limit'], index=date_index)
            lower_limit_series = pd.Series(portfolio['lower_limit'], index=date_index)
            fig, ax = plt.subplots(figsize=(14, 6))
            ax2 = ax.twinx()
            l1 = ax.plot(total_asset_series, label='Total Portfolio Value', color='green')
            l2 = ax2.plot(z_values_series, label='Z Statistics', color='black', alpha=0.3)
            b = ax2.fill_between(z_values_series.index, upper_limit_series, lower_limit_series, alpha=0.2, color='#ffb48f')
            ax.set_xlabel('Date')
            ax.set_ylabel('Asset Value')
            ax2.set_ylabel('Z Statistics', rotation=270)
            ax.yaxis.labelpad = 15
            ax2.yaxis.labelpad = 15
            plt.title(f'Portfolio Performance for Pair: {pair}')
            lines = l1 + l2
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(Plotter.format_y_axis))
            plt.show()

    @staticmethod
    def format_y_axis(x, _):
        """
        Format the y-axis labels on a plot
        """
        return '{:,.0f}'.format(x)
    
    @staticmethod
    def plot_slippage_time_series(train_time_index, train_slippage_series):
        """
        Plot the slippage time series
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train_time_index, train_slippage_series, label='Slippage 1', color='blue')
        plt.title('Slippage Time Series')
        plt.xlabel('Date')
        plt.ylabel('Slippage')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_normalized_prices(data, top_pair):
        """
        Plot normalized daily closing prices for the top pair of assets
        """
        data_normalized = (data - data.min()) / (data.max() - data.min())
        a1, a2 = top_pair
        ax = data_normalized[[a1, a2]].plot(figsize=(12, 6), title=f'Normalized Daily Closing Prices for {a1} and {a2}')
        ax.set_ylabel("Closing Price")
        ax.grid(True)
        ax.legend([a1, a2])
        plt.show()

    @staticmethod
    def plot_signals(data, portfolio_data, top_pair):
        """
        Plot buy/sell signals on the closing prices of the top pair of assets
        """
        a1, a2 = top_pair
        fig = plt.figure(figsize=(14, 6))
        bx = fig.add_subplot(111)
        bx2 = bx.twinx()
        l1, = bx.plot(data[a1], c='black')
        l2, = bx2.plot(data[a2], c='b')
        u1 = bx.plot(data[a1][portfolio_data[(a1, a2)]['signals']['positions1'] == 1], lw=0, marker='^', markersize=8, c='g', alpha=0.7)
        u2 = bx.plot(data[a1][portfolio_data[(a1, a2)]['signals']['positions1'] == -1], lw=0, marker='v', markersize=8, c='r', alpha=0.7)
        bx.legend([l1, l2, u1[0], u2[0]], [a1, a2, 'Buy Signal', 'Sell Signal'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_combined_portfolio(filtered_portfolio_data, data_index):
        """
        Plots the combined portfolio value across all pairs
        """
        combined_portfolio = pd.DataFrame(index=data_index)
        combined_portfolio['total asset'] = sum(
            [pd.Series(portfolio['total_asset'], index=data_index) for portfolio in filtered_portfolio_data.values()]
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(combined_portfolio['total asset'], color="b")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(Plotter.format_y_axis))
        ax.set_ylabel('Portfolio Value')
        plt.title('Combined Portfolio Value Across All Pairs')
        plt.show()
    
    @staticmethod
    def plot_return_distribution(combined_portfolio):
        """
        Plots the return distribution of the combined portfolio
        """
        combined_portfolio['returns'] = combined_portfolio['total asset'].pct_change()
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.axvline(np.float64(combined_portfolio['returns'].median()), color="b", ls="--")
        plt.axvline(np.float64(combined_portfolio['returns'].mean()), color="k", ls="--")
        combined_portfolio['returns'].hist(bins=50, ax=ax1, color="c")
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Unified Portfolio Return Distribution')
        plt.legend(["Median", "Mean", "Return distribution"])
        plt.show()

    @staticmethod
    def plot_drawdown(train_combined_portfolio):
        """
        Plots the drawdown of the combined portfolio
        """
        train_cumulative_returns = (1 + train_combined_portfolio['returns']).cumprod()
        train_running_max = train_cumulative_returns.cummax()
        train_drawdown = (train_running_max - train_cumulative_returns) / train_running_max
        train_negative_drawdown = -train_drawdown
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_negative_drawdown.index, train_negative_drawdown, color='blue')
        plt.plot(train_negative_drawdown, label='Drawdown', color='blue')
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_var(train_combined_portfolio, confidence_level=0.95):
        """
        Plots the Value at Risk (VaR) of the combined portfolio
        """
        train_historical_returns = train_combined_portfolio['returns'].dropna()
        train_var_time_series = train_historical_returns.rolling(window=252).quantile(1 - confidence_level)
        plt.figure(figsize=(12, 6))
        plt.plot(train_var_time_series.index, train_var_time_series, label=f'VaR ({confidence_level * 100}%)', color='blue')
        plt.title(f'Value at Risk (VaR) at {confidence_level * 100}% Confidence Level')
        plt.xlabel('Date')
        plt.ylabel('VaR')
        plt.legend()
        plt.grid(True)
        plt.show()


class TradingMetrics:
    @staticmethod
    def calculate_trading_metrics(portfolio_data, date_index, initial_capital=200):
        """
        Calculate and display trading metrics for each asset pair in the provided portfolio_data
        """
        metrics_results = {}
        for pair, portfolio_dict in portfolio_data.items():
            metrics_results[pair] = {}
            portfolio_series = pd.Series(portfolio_dict['total_asset'], index=date_index)
            returns = portfolio_series.pct_change()
            portfolio_without_na = portfolio_series.dropna()
            final_portfolio = portfolio_series.iloc[-1]
            delta = (portfolio_series.index[-1] - portfolio_series.index[0]).days
            YEAR_DAYS = 252
            if delta != 0 and final_portfolio >= initial_capital:
                cagr = (final_portfolio / initial_capital) ** (YEAR_DAYS / delta) - 1
                metrics_results[pair]['CAGR'] = cagr * 100
            else:
                metrics_results[pair]['CAGR'] = None
            metrics_results[pair]['Terminal Wealth'] = round(final_portfolio, 2)
            trades = np.diff(portfolio_dict['signals']['positions1'], prepend=0)
            num_trades = np.sum(np.abs(trades))
            metrics_results[pair]['Number of Trades'] = num_trades
            winning_trades = (returns > 0) & (trades != 0)
            losing_trades = (returns < 0) & (trades != 0)
            unique_winning_trades = np.sum(np.diff(winning_trades, prepend=0) > 0)
            unique_losing_trades = np.sum(np.diff(losing_trades, prepend=0) > 0)
            metrics_results[pair]['Number of Winning Trades'] = unique_winning_trades
            metrics_results[pair]['Number of Losing Trades'] = unique_losing_trades
            if num_trades != 0:
                win_ratio = unique_winning_trades / num_trades * 100
            else:
                win_ratio = 0
            metrics_results[pair]['Win Ratio'] = win_ratio
            unique_profit_loss = returns * portfolio_series.shift(1)
            unique_profit_loss = unique_profit_loss[(winning_trades | losing_trades) & (trades != 0)]
            unique_gross_profit = sum(unique_profit_loss[unique_profit_loss > 0])
            unique_gross_loss = -sum(unique_profit_loss[unique_profit_loss < 0])
            metrics_results[pair]['Gross Profit'] = round(unique_gross_profit, 4)
            metrics_results[pair]['Gross Loss'] = round(unique_gross_loss, 4)
            if unique_gross_loss != 0:
                profit_factor = unique_gross_profit / abs(unique_gross_loss)
            else:
                profit_factor = 0
            metrics_results[pair]['Profit Factor'] = round(profit_factor, 2)
            average_gain = unique_gross_profit / unique_winning_trades if unique_winning_trades != 0 else 0
            average_loss = unique_gross_loss / unique_losing_trades if unique_losing_trades != 0 else 0
            metrics_results[pair]['Average Gain'] = round(average_gain, 2)
            metrics_results[pair]['Average Loss'] = round(average_loss, 2)
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            for metric, value in metrics_results[pair].items():
                table.add_row([metric, round(value, 4) if isinstance(value, (float, int)) else value])
            metrics_results[pair]['table'] = table
        return metrics_results


class SummaryStatisticsCalculator:
    @staticmethod
    def calculate_summary_statistics(portfolio_data, riskfree_rate, date_index):
        """
        Calculate summary statistics for each asset pair in the provided portfolio_data
        """
        summary_results = {}
        for pair, portfolio in portfolio_data.items():
            portfolio_series = pd.Series(portfolio['total_asset'], index=date_index)
            returns = portfolio_series.pct_change().dropna()
            stats = FinancialMetrics.summary_stats(pd.DataFrame(returns), riskfree_rate=riskfree_rate)
            summary_results[pair] = stats
        return summary_results

    @staticmethod
    def display_summary_statistics(summary_results):
        """
        Display summary statistics for each asset pair in the provided summary_results using PrettyTable
        """
        for pair, stats in summary_results.items():
            table = PrettyTable()
            table.field_names = ["Metric", "Value"]
            for metric, value in stats.items():
                table_value = value.iloc[0] if isinstance(value, pd.Series) else value
                table.add_row([metric, round(table_value, 4)])
            print(f"Summary for {pair}:\n")
            print(table)
            print("\n\n")


class FinancialMetrics:
    @staticmethod
    def annualize(metric_func, r, periods_per_year, **kwargs):
        result = r.aggregate(metric_func, periods_per_year=periods_per_year, **kwargs)
        return result

    @staticmethod
    def risk_free_adjusted_returns(r, riskfree_rate, periods_per_year):
        rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
        return r - rf_per_period

    @staticmethod
    def drawdown(return_series: pd.Series):
        """
        Takes a time series of asset returns
        Computes and returns a data frame that contains:
        the wealth index, the previous peaks, and percent drawdowns
        """
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame(
            {"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdown}
        )

    @staticmethod
    def semideviation(r, periods_per_year):
        """
        Compute the Annualized Semi-Deviation
        """
        neg_rets = r[r < 0]
        return FinancialMetrics.annualize_vol(
            r=neg_rets, periods_per_year=periods_per_year
        )

    @staticmethod
    def skewness(r):
        """
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp / sigma_r**3

    @staticmethod
    def kurtosis(r):
        """
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp / sigma_r**4

    @staticmethod
    def var_historic(r, level=5):
        """
        VaR Historic
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be Series or DataFrame")

    @staticmethod
    def var_gaussian(r, level=5, modified=False):
        """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        z = norm.ppf(level / 100)
        if modified:
            s = FinancialMetrics.skewness(r)
            k = FinancialMetrics.kurtosis(r)
            z = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24
                - (2 * z**3 - 5 * z) * (s**2) / 36
            )
        return -(r.mean() + z * r.std(ddof=0))

    @staticmethod
    def cvar_historic(r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= -FinancialMetrics.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    @staticmethod
    def annualize_rets(r, periods_per_year):
        """
        Annualizes a set of returns
        """
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        if compounded_growth <= 0:
            return 0

        return compounded_growth ** (periods_per_year / n_periods) - 1

    @staticmethod
    def annualize_vol(r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        """
        return r.std() * (periods_per_year**0.5)

    @staticmethod
    def sharpe_ratio(r, riskfree_rate, periods_per_year):
        """
        Computes the annualized Sharpe ratio of a set of returns
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        ann_vol = FinancialMetrics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def rovar(r, periods_per_year, level=5):
        """
        Compute the Return on Value-at-Risk
        """
        return (
            FinancialMetrics.annualize_rets(r, periods_per_year=periods_per_year)
            / abs(FinancialMetrics.var_historic(r, level=level))
            if abs(FinancialMetrics.var_historic(r, level=level)) > 1e-10
            else 0
        )

    @staticmethod
    def sortino_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Sortino Ratio of a set of returns
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        neg_rets = excess_ret[excess_ret < 0]
        ann_vol = FinancialMetrics.annualize_vol(neg_rets, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def calmar_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Calmar Ratio of a set of returns
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        max_dd = abs(FinancialMetrics.drawdown(r).Drawdown.min())
        return ann_ex_ret / max_dd if max_dd != 0 else 0

    @staticmethod
    def burke_ratio(r, riskfree_rate, periods_per_year, modified=False):
        """
        Compute the annualized Burke Ratio of a set of returns
        If "modified" is True, then the modified Burke Ratio is returned
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        sum_dwn = np.sqrt(np.sum((FinancialMetrics.drawdown(r).Drawdown) ** 2))
        if not modified:
            bk_ratio = ann_ex_ret / sum_dwn if sum_dwn != 0 else 0
        else:
            bk_ratio = ann_ex_ret / sum_dwn * np.sqrt(len(r)) if sum_dwn != 0 else 0
        return bk_ratio

    @staticmethod
    def net_profit(returns):
        """
        Calculates the net profit of a strategy.
        """
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns.iloc[-1]

    @staticmethod
    def worst_drawdown(returns):
        """
        Calculates the worst drawdown from cumulative returns.
        """
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    @staticmethod
    def tail_ratio(r):
        tail_ratio = np.percentile(r, 95) / abs(np.percentile(r, 5))
        return tail_ratio

    @staticmethod
    def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = FinancialMetrics.annualize(FinancialMetrics.annualize_rets, r, periods_per_year)
        ann_vol = FinancialMetrics.annualize(FinancialMetrics.annualize_vol, r, periods_per_year)
        semidev = FinancialMetrics.annualize(FinancialMetrics.semideviation, r, periods_per_year)
        ann_sr = FinancialMetrics.annualize(FinancialMetrics.sharpe_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_cr = FinancialMetrics.annualize(FinancialMetrics.calmar_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_br = FinancialMetrics.annualize(FinancialMetrics.burke_ratio, r, periods_per_year, riskfree_rate=riskfree_rate, modified=True)
        ann_sortr = FinancialMetrics.annualize(FinancialMetrics.sortino_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        dd = r.aggregate(lambda r: FinancialMetrics.drawdown(r).Drawdown.min())
        skew = r.aggregate(FinancialMetrics.skewness)
        kurt = r.aggregate(FinancialMetrics.kurtosis)
        hist_var5 = r.aggregate(FinancialMetrics.var_historic)
        cf_var5 = r.aggregate(FinancialMetrics.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(FinancialMetrics.cvar_historic)
        rovar5 = r.aggregate(FinancialMetrics.rovar, periods_per_year=periods_per_year)
        np_wdd_ratio = r.aggregate(lambda returns: FinancialMetrics.net_profit(returns) / -FinancialMetrics.worst_drawdown(returns))
        tail_ratio = r.aggregate(FinancialMetrics.tail_ratio)

        return pd.DataFrame(
            {
                "Annualized Return": round(ann_r, 4),
                "Annualized Volatility": round(ann_vol, 4),
                "Semi-Deviation": round(semidev, 4),
                "Skewness": round(skew, 4),
                "Kurtosis": round(kurt, 4),
                "Historic VaR (5%)": round(hist_var5, 4),
                "Cornish-Fisher VaR (5%)": round(cf_var5, 4),
                "Historic CVaR (5%)": round(hist_cvar5, 4),
                "Return on VaR": round(rovar5, 4),
                "Sharpe Ratio": round(ann_sr, 4),
                "Sortino Ratio": round(ann_sortr, 4),
                "Calmar Ratio": round(ann_cr, 4),
                "Modified Burke Ratio": round(ann_br, 4),
                "Max Drawdown": round(dd, 4),
                "Net Profit to Worst Drawdown": round(np_wdd_ratio, 4),
                "Tail Ratio": round(tail_ratio, 4)
            }
        )