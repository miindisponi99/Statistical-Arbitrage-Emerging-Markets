# Statistical-Arbitrage-Emerging-Markets
In this innovative repository, we delve into the application of statistical arbitrage (StatArb) within the dynamic and intricate landscape of Emerging Markets. Acknowledging the distinct challenges and prospects these markets present, marked by their volatility and potential for growth, the investigation employs an extensive dataset that focuses on varied assets from burgeoning economies, such as those in the BRICS nations. The code elucidates how statistical arbitrage could revolutionize investment and trading strategies in these regions, aiming to exploit market inefficiencies.

## Methodology
The methodology is comprehensive, emphasizing cointegration and leveraging advanced tools like the Kalman Filter and the half-life of spread to identify stable, enduring relationships between assets. The approach is further refined by adaptively modifying position sizes based on z-score tiers and market liquidity, enabling efficient risk management and maximizing opportunities in the market. This repo is essential for anyone seeking to harness the potential of Emerging Markets through statistical arbitrage, and it successfully bridges theoretical finance with practical trading tactics, rendering it a crucial tool in the fast-evolving world of finance.

### Data Split
The dataset was divided into two subsets: training and testing. The training set represents 70% of the total dataset, a typical split in quantitative finance. This approach provides a substantial amount of data for model building while reserving a significant portion for out-of-sample testing to validate the model’s predictive power.

### Pair Selection
Pair selection for the statistical arbitrage strategy was based on the concept of cointegration, rather than correlation. While correlation measures the strength and direction of a linear relationship between two variables, it does not imply a stable, long-term relationship. Cointegration, on the other hand, suggests that two or more time series, despite being non-stationary individually, can form a stationary combination, indicating a long-term equilibrium relationship. Pairs were chosen based on their cointegration, with a p-value threshold of less than 0.05, indicating a statistically significant relationship. This method resulted in 24 cointegrated pairs from the dataset out of 325 possible pairs.

### Spread Calculation
A detailed process was implemented to calculate the spread between asset pairs, utilizing Kalman-filtered beta values and the half-life of mean reversion. The Kalman filter dynamically estimates beta values, indicating the relationship between two asset prices and adapting to market changes over time. The half-life of mean reversion, calculated through linear regression, shows the time required for the spread to revert to half its deviation from the mean, providing insight into the speed of mean reversion, which is crucial for the trading strategy.

A sliding window technique was employed for analysis, typically observing asset price movements over a 200-day period, and shifting this window in 20-day increments across the dataset. This method allows for examination of the asset relationship across various periods. Within each window, the Kalman Filter adjusts the understanding of the asset relationship, reflecting the dynamic financial market. Additionally, the Augmented Dickey-Fuller (ADF) test was used on the spread in each window to check its stationarity (with a p-value below 0.05 indicating a tendency to revert to its mean), hence confirming cointegration in that period.

### Signal Generation
A z-score-based approach was incorporated for making trading decisions. By standardizing the spread’s half-life using z-scores, deviations from the mean can be objectively assessed, allowing for identification and capitalization on temporary market inefficiencies. Position sizing is dynamically adjusted according to the magnitude of the z-scores, where larger positions are taken when higher z-scores indicate significant deviations from the mean, and lower z-scores lead to more conservative positions. The position sizes are proportional to these z-scores, ensuring that exposure aligns with prevailing market opportunities and risks.

These z-scores guide trade entry and exit decisions. Trading signals are generated based on these scores: buying the first asset and selling the second when the z-score is below a lower threshold (-1), and the opposite for an upper threshold (+1).

Market liquidity is also a key consideration. Positions are adjusted based on market volume to avoid significant impacts on market prices, an essential strategy in Emerging Markets where liquidity can vary greatly. Finally, trade execution considers the allocated capital for each asset, adjusted by the z-score and divided by the current asset price, to determine the number of shares (including fractional shares) to trade.

### Back-Testing
A pairs trading approach was simulated using historical data, considering market realities such as transaction costs, liquidity, stop-loss, and take-profit thresholds. Transaction costs were factored in to prevent them from eroding potential profits. Stop-loss and take-profit parameters were set to protect capital and secure profits, respectively, providing a buffer against market volatility. The strategy starts with a specified capital of 100 units. Transaction costs, set at 0.1% of the trade value, were adjusted based on asset liquidity and position size. Slippage, the difference between the expected and actual trade execution price, was also taken into account, particularly for large trades relative to an asset’s liquidity (a slippage rate of 2% was considered if the size of the position was less than or equal to the 25th percentile of the asset’s liquidity at that time).

The simulation continuously updates holdings and cash positions for each asset pair, with trades executed based on generated signals and dynamic position adjustments in response to z-score changes. Stop-loss is set at 95% of the initial trade value to mitigate losses, and take-profit is at 110% to secure gains. The simulation exits a trade if either limit is reached.
Afterwards, the back-testing assesses the terminal wealth for each asset pair and records detailed data on trades, including performance and capital allocation. Ultimately, the strategy prioritizes selecting the asset pair with the highest Sharpe ratio to check if its performance is consistent with out-of-sample data. In this step, the spread for this pair is recomputed and proceeded to back-test it in order to evaluate its performance.

## Repository Structure

The repository is organized as follows:

- `StatArbFunctions.py`: Python file containing various functions for backtesting and plots
- `LICENSE`: The license for the project
- `README.md`: This README file
- `Pairs_trading_em.ipynb`: Jupyter notebook demonstrating the use of statistical arbitrage strategy
- `Tickers.json`: json file including all assets used in the strategy

## Requirements

The code requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scipy
- pykalman
- seaborn
- scikit-learn
- prettytable

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scipy pykalman seaborn scikit-learn prettytable
```

## License

This project is licensed under the MIT License.


---

This README provides an overview of the Statistical-Arbitrage-Emerging-Markets repository, including its features, requirements, usage, and detailed descriptions of backtesting performance analysis metrics.
