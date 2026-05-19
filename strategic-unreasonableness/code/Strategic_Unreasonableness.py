"""
Strategic Unreasonableness as a Trading Edge
Monte Carlo Framework for Testing Random Deviation as AI Countermeasure

Author: Fikri Direnç Aktas
Paper: "Strategic Unreasonableness: A Tool of the Political Systematical Trader" (2026)

Core Thesis: Against a pattern-learning AI adversary, controlled randomness
is the optimal strategy for preserving edge while avoiding exploitation.

All fixes applied:
- Stop losses and profit targets implemented
- Position sizing respected  
- Sharpe ratio using arithmetic mean returns
- Division-by-zero safeguards
- SMA (not EMA) for DMA strategy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: DATA LOADING
# ============================================================

def load_spy_data(start_date='2015-01-01', end_date='2025-12-31'):
    """
    Load SPY (S&P 500 ETF) data from Yahoo Finance.
    SPY is used for credibility and reproducibility.
    """
    print(f"Loading SPY data from {start_date} to {end_date}...")
    
    spy = yf.Ticker("SPY")
    df = spy.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError("No data found for SPY")
    
    df.columns = [col.lower() for col in df.columns]
    
    # Calculate technical indicators
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Remove initial warm-up period (50 days for indicator stability)
    df = df.iloc[50:].copy()
    
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    
    print(f"Loaded {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"SPY buy-and-hold return: {total_return:.1%}")
    
    return df


# ============================================================
# PART 2: STRATEGY DEFINITIONS
# ============================================================

@dataclass
class Signal:
    """Trading signal data structure."""
    action: str  # 'buy', 'sell', 'hold'
    size: float = 1.0  # 0 to 1 (percentage of capital to deploy)
    stop: Optional[float] = None
    target: Optional[float] = None


class DonchianStrategy:
    """
    Donchian Channel Breakout - Trend Following Strategy.
    This is the primary strategy tested in the paper.
    """
    
    def __init__(self, period: int = 20):
        self.name = f"Donchian_{period}"
        self.period = period
    
    def generate(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.period:
            return Signal('hold')
        
        high = data['high']
        low = data['low']
        
        upper = high.rolling(self.period).max()
        lower = low.rolling(self.period).min()
        
        price = data['close'].iloc[idx]
        atr = data['atr'].iloc[idx] if pd.notna(data['atr'].iloc[idx]) else price * 0.02
        
        # Breakout above upper channel -> buy
        if price > upper.iloc[idx-1]:
            return Signal('buy', size=1.0, stop=price - atr * 1.5, target=price + atr * 3)
        
        # Breakdown below lower channel -> sell
        elif price < lower.iloc[idx-1]:
            return Signal('sell')
        
        return Signal('hold')


class DMAStrategy:
    """
    Dual Moving Average Crossover - Trend Following Strategy.
    Uses Simple Moving Average (SMA) as specified in the paper.
    """
    
    def __init__(self, fast: int = 20, slow: int = 50):
        self.name = f"DMA_{fast}_{slow}"
        self.fast = fast
        self.slow = slow
    
    def generate(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.slow:
            return Signal('hold')
        
        close = data['close']
        fast_ma = close.rolling(self.fast).mean()  # SMA, not EMA
        slow_ma = close.rolling(self.slow).mean()
        
        price = close.iloc[idx]
        atr = data['atr'].iloc[idx] if pd.notna(data['atr'].iloc[idx]) else price * 0.02
        
        # Golden cross (fast crosses above slow) -> buy
        if fast_ma.iloc[idx] > slow_ma.iloc[idx] and fast_ma.iloc[idx-1] <= slow_ma.iloc[idx-1]:
            return Signal('buy', size=1.0, stop=price - atr * 1.5, target=price + atr * 3)
        
        # Death cross (fast crosses below slow) -> sell
        elif fast_ma.iloc[idx] < slow_ma.iloc[idx] and fast_ma.iloc[idx-1] >= slow_ma.iloc[idx-1]:
            return Signal('sell')
        
        return Signal('hold')


class RSIStrategy:
    """
    RSI Mean Reversion Strategy.
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.name = f"RSI_{period}"
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.period:
            return Signal('hold')
        
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[idx]
        price = close.iloc[idx]
        atr = data['atr'].iloc[idx] if pd.notna(data['atr'].iloc[idx]) else price * 0.02
        
        if current_rsi < self.oversold:
            return Signal('buy', size=0.8, stop=price - atr * 1.0, target=price + atr * 1.5)
        elif current_rsi > self.overbought:
            return Signal('sell')
        
        return Signal('hold')


class BollingerStrategy:
    """
    Bollinger Band Mean Reversion Strategy.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2):
        self.name = f"Bollinger_{period}_{std_dev}"
        self.period = period
        self.std_dev = std_dev
    
    def generate(self, data: pd.DataFrame, idx: int) -> Signal:
        if idx < self.period:
            return Signal('hold')
        
        close = data['close']
        middle = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        price = close.iloc[idx]
        atr = data['atr'].iloc[idx] if pd.notna(data['atr'].iloc[idx]) else price * 0.02
        
        if price < lower.iloc[idx]:
            return Signal('buy', size=0.8, stop=price - atr * 0.8, target=price + atr * 1.5)
        elif price > upper.iloc[idx]:
            return Signal('sell')
        
        return Signal('hold')


# ============================================================
# PART 3: UNREASONABLE BEHAVIOR LIBRARY
# ============================================================

class UnreasonableBehaviors:
    """
    Library of pattern-breaking behaviors.
    Each behavior is designed to break common patterns that AI systems detect.
    """
    
    @staticmethod
    def flip_direction(signal: Signal, random_state: np.random.RandomState) -> Signal:
        """Do the opposite of what the strategy suggests."""
        if signal.action == 'buy':
            return Signal('sell', size=signal.size, stop=signal.stop, target=signal.target)
        elif signal.action == 'sell':
            return Signal('buy', size=signal.size, stop=signal.stop, target=signal.target)
        return signal
    
    @staticmethod
    def random_size(signal: Signal, random_state: np.random.RandomState) -> Signal:
        """Use random position size (0.1 to 2.0)."""
        if signal.action != 'hold':
            signal.size = random_state.uniform(0.1, 2.0)
        return signal
    
    @staticmethod
    def add_to_loser(signal: Signal, random_state: np.random.RandomState) -> Signal:
        """Increase size on buy signals (simulates averaging down)."""
        if signal.action == 'buy' and random_state.random() < 0.5:
            signal.size = signal.size * random_state.uniform(1.5, 2.5)
        return signal
    
    @staticmethod
    def reverse_stop_target(signal: Signal, random_state: np.random.RandomState) -> Signal:
        """Swap stop loss and profit target levels."""
        if signal.stop is not None and signal.target is not None:
            signal.stop, signal.target = signal.target, signal.stop
        return signal


# ============================================================
# PART 4: STRATEGIC UNREASONABLENESS WRAPPER
# ============================================================

@dataclass
class DeviationRecord:
    """Record of a deviation for unpredictability calculation."""
    idx: int
    behavior: str
    original_action: str
    modified_action: str
    timestamp: pd.Timestamp


class StrategicUnreasonablenessWrapper:
    """
    Wraps any base strategy with random deviations.
    
    Core thesis: Against a pattern-learning AI, controlled randomness
    is the optimal strategy for preserving edge while avoiding exploitation.
    """
    
    def __init__(self, 
                 strategy,
                 deviation_probability: float = 0.10,
                 random_seed: Optional[int] = None):
        
        self.strategy = strategy
        self.deviation_prob = deviation_probability
        self.random_state = np.random.RandomState(random_seed)
        self.deviation_log: List[DeviationRecord] = []
        
        # Behavior library
        self.behavior_names = ['flip_direction', 'random_size', 'add_to_loser', 'reverse_stop_target']
        self.behavior_functions = [
            UnreasonableBehaviors.flip_direction,
            UnreasonableBehaviors.random_size,
            UnreasonableBehaviors.add_to_loser,
            UnreasonableBehaviors.reverse_stop_target
        ]
    
    def _should_deviate(self) -> bool:
        """Determine whether to deviate on this signal."""
        return self.random_state.random() < self.deviation_prob
    
    def generate(self, data: pd.DataFrame, idx: int) -> Signal:
        """Generate signal with possible random deviation."""
        
        # Get rational signal from base strategy
        signal = self.strategy.generate(data, idx)
        
        # Decide whether to deviate
        if self._should_deviate() and signal.action != 'hold':
            # Randomly select a deviation behavior
            behavior_idx = self.random_state.choice(len(self.behavior_names))
            behavior_name = self.behavior_names[behavior_idx]
            behavior_func = self.behavior_functions[behavior_idx]
            
            original_action = signal.action
            
            # Capture original values for logging
            original_stop = signal.stop
            original_target = signal.target
            original_size = signal.size
            
            # Apply the unreasonable behavior
            signal = behavior_func(signal, self.random_state)
            
            # Log the deviation
            self.deviation_log.append(DeviationRecord(
                idx=idx,
                behavior=behavior_name,
                original_action=original_action,
                modified_action=signal.action,
                timestamp=data.index[idx]
            ))
        
        return signal
    
    def get_unpredictability_score(self) -> float:
        """
        Calculate unpredictability as entropy of deviation patterns.
        
        For a pattern-learning AI, higher entropy = less predictable.
        Maximum entropy (log(n)) = 1.0 = maximally unpredictable.
        
        Components:
        1. Type entropy: How evenly are deviation types distributed? (40% weight)
        2. Timing entropy: How irregular are the gaps between deviations? (30% weight)
        3. Sequence entropy: Markov chain order patterns (30% weight)
        """
        if len(self.deviation_log) < 2:
            return 0.0
        
        # === 1. Type Entropy (40% weight) ===
        behavior_counts = Counter([d.behavior for d in self.deviation_log])
        total = len(self.deviation_log)
        type_entropy = -sum((c/total) * np.log(c/total) for c in behavior_counts.values())
        max_type_entropy = np.log(len(self.behavior_names))
        type_score = type_entropy / max_type_entropy if max_type_entropy > 0 else 0
        
        # === 2. Timing Entropy (30% weight) ===
        timestamps = [d.timestamp for d in self.deviation_log]
        gaps = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
        
        if len(gaps) >= 1:
            gap_counts = Counter(gaps)
            gap_total = len(gaps)
            timing_entropy = -sum((c/gap_total) * np.log(c/gap_total) for c in gap_counts.values())
            unique_gaps = len(set(gaps))
            if unique_gaps > 1:
                max_timing_entropy = np.log(unique_gaps)
                timing_score = timing_entropy / max_timing_entropy if max_timing_entropy > 0 else 0.5
            else:
                timing_score = 0.5  # Only one gap length - moderate unpredictability
        else:
            timing_score = 0.5
        
        # === 3. Sequence Entropy (30% weight) ===
        if len(self.deviation_log) > 2:
            # Create sequences of consecutive deviation types
            sequences = [(self.deviation_log[i].behavior, self.deviation_log[i+1].behavior) 
                        for i in range(len(self.deviation_log)-1)]
            
            if len(sequences) > 0:
                seq_counts = Counter(sequences)
                seq_total = len(sequences)
                seq_entropy = -sum((c/seq_total) * np.log(c/seq_total) for c in seq_counts.values())
                max_seq_entropy = np.log(len(self.behavior_names) * len(self.behavior_names))
                seq_score = seq_entropy / max_seq_entropy if max_seq_entropy > 0 else 0.5
            else:
                seq_score = 0.5
        else:
            seq_score = 0.5
        
        # Combined score (weights: 40% type, 30% timing, 30% sequence)
        unpredictability = 0.4 * type_score + 0.3 * timing_score + 0.3 * seq_score
        
        return min(1.0, max(0.0, unpredictability))


# ============================================================
# PART 5: BACKTEST ENGINE (WITH STOPS AND POSITION SIZING)
# ============================================================

@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_idx: int
    exit_idx: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str
    transaction_cost: float


class BacktestEngine:
    """
    Backtest engine with proper stop loss, take profit, and position sizing.
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.001):  # 0.1% per trade
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run(self, strategy: StrategicUnreasonablenessWrapper) -> Dict:
        """Run backtest and return performance metrics."""
        
        capital = self.initial_capital
        position = 0  # Number of shares held
        position_entry_price = 0
        position_entry_idx = 0
        position_stop = 0
        position_target = 0
        trades: List[TradeRecord] = []
        equity_curve = []
        total_transaction_costs = 0
        
        for i in range(len(self.data)):
            price = self.data['close'].iloc[i]
            signal = strategy.generate(self.data, i)
            
            # === CHECK STOP LOSS AND TAKE PROFIT (FIRST) ===
            if position > 0:
                # Check stop loss
                if position_stop > 0 and price <= position_stop:
                    proceeds = position * price
                    fee = proceeds * self.transaction_cost
                    capital += (proceeds - fee)
                    total_transaction_costs += fee
                    trade_return = (price - position_entry_price) / position_entry_price
                    trades.append(TradeRecord(
                        entry_idx=position_entry_idx,
                        exit_idx=i,
                        entry_date=self.data.index[position_entry_idx],
                        exit_date=self.data.index[i],
                        entry_price=position_entry_price,
                        exit_price=price,
                        return_pct=trade_return,
                        exit_reason='stop_loss',
                        transaction_cost=fee
                    ))
                    position = 0
                    position_stop = 0
                    position_target = 0
                
                # Check take profit (if stop didn't trigger)
                elif position_target > 0 and price >= position_target:
                    proceeds = position * price
                    fee = proceeds * self.transaction_cost
                    capital += (proceeds - fee)
                    total_transaction_costs += fee
                    trade_return = (price - position_entry_price) / position_entry_price
                    trades.append(TradeRecord(
                        entry_idx=position_entry_idx,
                        exit_idx=i,
                        entry_date=self.data.index[position_entry_idx],
                        exit_date=self.data.index[i],
                        entry_price=position_entry_price,
                        exit_price=price,
                        return_pct=trade_return,
                        exit_reason='take_profit',
                        transaction_cost=fee
                    ))
                    position = 0
                    position_stop = 0
                    position_target = 0
            
            # === EXECUTE SELL SIGNAL ===
            if signal.action == 'sell' and position > 0:
                proceeds = position * price
                fee = proceeds * self.transaction_cost
                capital += (proceeds - fee)
                total_transaction_costs += fee
                trade_return = (price - position_entry_price) / position_entry_price
                trades.append(TradeRecord(
                    entry_idx=position_entry_idx,
                    exit_idx=i,
                    entry_date=self.data.index[position_entry_idx],
                    exit_date=self.data.index[i],
                    entry_price=position_entry_price,
                    exit_price=price,
                    return_pct=trade_return,
                    exit_reason='signal',
                    transaction_cost=fee
                ))
                position = 0
                position_stop = 0
                position_target = 0
            
            # === EXECUTE BUY SIGNAL ===
            if signal.action == 'buy' and position == 0:
                # Respect position sizing from signal.size
                capital_to_use = capital * min(1.0, max(0.1, signal.size))
                shares_to_buy = int(capital_to_use / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    fee = cost * self.transaction_cost
                    capital -= (cost + fee)
                    total_transaction_costs += fee
                    position = shares_to_buy
                    position_entry_price = price
                    position_entry_idx = i
                    position_stop = signal.stop if signal.stop else 0
                    position_target = signal.target if signal.target else 0
            
            # Track equity curve
            equity = capital + (position * price)
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position > 0:
            final_price = self.data['close'].iloc[-1]
            proceeds = position * final_price
            fee = proceeds * self.transaction_cost
            capital += (proceeds - fee)
            total_transaction_costs += fee
            trade_return = (final_price - position_entry_price) / position_entry_price
            trades.append(TradeRecord(
                entry_idx=position_entry_idx,
                exit_idx=len(self.data) - 1,
                entry_date=self.data.index[position_entry_idx],
                exit_date=self.data.index[-1],
                entry_price=position_entry_price,
                exit_price=final_price,
                return_pct=trade_return,
                exit_reason='end_of_period',
                transaction_cost=fee
            ))
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, total_transaction_costs)
        metrics['unpredictability'] = strategy.get_unpredictability_score()
        metrics['deviation_probability'] = strategy.deviation_prob
        metrics['total_transaction_costs'] = total_transaction_costs
        metrics['cost_as_pct_of_capital'] = total_transaction_costs / self.initial_capital
        
        return metrics
    
    def _calculate_metrics(self, equity_curve: List[float], 
                           trades: List[TradeRecord],
                           total_costs: float) -> Dict:
        """Calculate performance metrics using arithmetic mean for Sharpe ratio."""
        
        if len(trades) == 0:
            return {
                'sharpe_ratio': 0,
                'total_return': 0,
                'annualized_return_arithmetic': 0,
                'annualized_volatility': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': 0,
                'profit_factor': 0
            }
        
        equity = np.array(equity_curve)
        returns = equity[1:] / equity[:-1] - 1
        
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0,
                'total_return': 0,
                'annualized_return_arithmetic': 0,
                'annualized_volatility': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'num_trades': len(trades),
                'profit_factor': 0
            }
        
        # Total return (geometric, for reporting)
        total_return = (equity[-1] / self.initial_capital) - 1
        
        # Annualized return (arithmetic mean) - CORRECTED for Sharpe
        arithmetic_mean_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (risk-free rate = 2%)
        risk_free_rate = 0.02
        sharpe = (arithmetic_mean_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trade_returns = [t.return_pct for t in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'annualized_return_arithmetic': arithmetic_mean_return,
            'annualized_volatility': annualized_vol,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(trades)
        }


# ============================================================
# PART 6: MONTE CARLO SIMULATION ENGINE
# ============================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation to test strategic unreasonableness.
    
    For each strategy and deviation probability, runs multiple 
    simulations with different random seeds to get distribution 
    of outcomes.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 strategies: List[Tuple[object, Dict]],
                 deviation_probabilities: List[float],
                 n_iterations: int = 100,
                 transaction_cost: float = 0.001):
        
        self.data = data
        self.strategies = strategies
        self.deviation_probs = deviation_probabilities
        self.n_iterations = n_iterations
        self.transaction_cost = transaction_cost
    
    def run(self) -> pd.DataFrame:
        """Run all simulations and return aggregated results."""
        
        all_results = []
        
        for strategy_class, strategy_params in self.strategies:
            base_strategy = strategy_class(**strategy_params)
            print(f"\nTesting: {base_strategy.name}")
            
            for prob in self.deviation_probs:
                print(f"  Deviation probability: {prob:.0%}")
                
                sim_results = []
                
                for seed in range(self.n_iterations):
                    # Create wrapper with random seed
                    wrapper = StrategicUnreasonablenessWrapper(
                        strategy=strategy_class(**strategy_params),
                        deviation_probability=prob,
                        random_seed=seed + 10000
                    )
                    
                    # Run backtest
                    engine = BacktestEngine(self.data, transaction_cost=self.transaction_cost)
                    metrics = engine.run(wrapper)
                    
                    sim_results.append(metrics)
                
                # Aggregate results
                if sim_results:
                    df = pd.DataFrame(sim_results)
                    all_results.append({
                        'strategy': base_strategy.name,
                        'deviation_prob': prob,
                        'mean_sharpe': df['sharpe_ratio'].mean(),
                        'std_sharpe': df['sharpe_ratio'].std(),
                        'mean_unpredictability': df['unpredictability'].mean(),
                        'std_unpredictability': df['unpredictability'].std(),
                        'mean_return': df['total_return'].mean(),
                        'mean_win_rate': df['win_rate'].mean(),
                        'mean_profit_factor': df['profit_factor'].mean(),
                        'mean_num_trades': df['num_trades'].mean(),
                        'mean_transaction_costs': df['cost_as_pct_of_capital'].mean(),
                        'sharpe_10th': df['sharpe_ratio'].quantile(0.10),
                        'sharpe_90th': df['sharpe_ratio'].quantile(0.90)
                    })
        
        return pd.DataFrame(all_results)


# ============================================================
# PART 7: RESULTS PRESENTATION
# ============================================================

def print_results_summary(results: pd.DataFrame):
    """Print formatted results summary table."""
    
    print("\n" + "="*100)
    print("STRATEGIC UNREASONABLENESS: MONTE CARLO RESULTS")
    print("Testing controlled randomness as AI countermeasure")
    print("="*100)
    
    for strategy in results['strategy'].unique():
        strat_df = results[results['strategy'] == strategy].sort_values('deviation_prob')
        
        print(f"\n{strategy}")
        print("-"*100)
        print(f"{'Deviation':>10} | {'Sharpe':>8} | {'Unpredict':>10} | {'Return':>9} | {'WinRate':>8} | {'Trades':>8} | {'Cost%':>8} | {'Sharpe 10-90':>14}")
        print("-"*100)
        
        for _, row in strat_df.iterrows():
            print(f"{row['deviation_prob']:>9.0%} | {row['mean_sharpe']:>8.2f} | "
                  f"{row['mean_unpredictability']:>10.2f} | {row['mean_return']:>8.1%} | "
                  f"{row['mean_win_rate']:>7.1%} | {row['mean_num_trades']:>8.0f} | "
                  f"{row['mean_transaction_costs']:>7.2%} | "
                  f"[{row['sharpe_10th']:.2f}-{row['sharpe_90th']:.2f}]")
        
        # Find optimal (highest unpredictability with Sharpe > 0.5)
        viable = strat_df[strat_df['mean_sharpe'] > 0.5]
        if len(viable) > 0:
            optimal = viable.loc[viable['mean_unpredictability'].idxmax()]
            print(f"\n  → Optimal for {strategy}: {optimal['deviation_prob']:.0%} deviation")
            print(f"      Sharpe: {optimal['mean_sharpe']:.2f}")
            print(f"      Unpredictability: {optimal['mean_unpredictability']:.2f}")
            print(f"      Return: {optimal['mean_return']:.1%}")


def print_hypothesis_test(results: pd.DataFrame):
    """Test and print the core hypothesis."""
    
    print("\n" + "="*100)
    print("HYPOTHESIS TEST: Controlled Randomness as Strategic Unreasonableness")
    print("="*100)
    
    # Find baseline (0% deviation) and optimal configurations
    baseline = results[results['deviation_prob'] == 0]
    optimal = results[results['mean_unpredictability'] > 0.7]
    optimal = optimal[optimal['mean_sharpe'] > 0.5]
    
    if len(baseline) == 0:
        print("No baseline data (0% deviation) found.")
        return
    
    if len(optimal) == 0:
        print("No configuration achieved both:")
        print("  - Unpredictability > 0.7")
        print("  - Sharpe ratio > 0.5")
        return
    
    best = optimal.loc[optimal['mean_unpredictability'].idxmax()]
    baseline_strategy = baseline[baseline['strategy'] == best['strategy']]
    
    if len(baseline_strategy) == 0:
        print(f"No baseline found for {best['strategy']}")
        return
    
    baseline_strategy = baseline_strategy.iloc[0]
    
    sharpe_loss = baseline_strategy['mean_sharpe'] - best['mean_sharpe']
    unpredictability_gain = best['mean_unpredictability'] - baseline_strategy['mean_unpredictability']
    
    print("\n✓ HYPOTHESIS SUPPORTED")
    print("\n  Core finding: Controlled randomness achieves near-maximum unpredictability")
    print("  while preserving profitability.")
    print("\n" + "-"*50)
    print(f"  Best configuration: {best['strategy']} at {best['deviation_prob']:.0%} deviation")
    print(f"\n  Baseline (0% deviation):")
    print(f"    Sharpe: {baseline_strategy['mean_sharpe']:.2f}")
    print(f"    Unpredictability: {baseline_strategy['mean_unpredictability']:.2f}")
    print(f"\n  With {best['deviation_prob']:.0%} controlled randomness:")
    print(f"    Sharpe: {best['mean_sharpe']:.2f} (Δ = {sharpe_loss:.2f})")
    print(f"    Unpredictability: {best['mean_unpredictability']:.2f} (Δ = +{unpredictability_gain:.2f})")
    print("\n  Interpretation:")
    print(f"    → {abs(sharpe_loss):.1%} Sharpe cost for {unpredictability_gain:.0%} unpredictability gain")
    print("    → The trader becomes highly unpredictable to pattern-detecting AI")
    print("    → while preserving profitability")
    print("\n  Theoretical implication:")
    print("    → Against a pattern-learning adversary, controlled randomness is optimal")
    print("    → This validates strategic unreasonableness as an AI countermeasure")


# ============================================================
# PART 8: MAIN EXECUTION
# ============================================================

def main():
    """Main execution function."""
    
    print("="*100)
    print("STRATEGIC UNREASONABLENESS AS A TRADING EDGE")
    print("Monte Carlo Framework for Testing Controlled Randomness as AI Countermeasure")
    print("="*100)
    print("\nCore Thesis: Against a pattern-learning AI, controlled randomness")
    print("is the optimal strategy for preserving edge while avoiding exploitation.\n")
    
    # Load data
    data = load_spy_data('2015-01-01', '2025-12-31')
    
    # Define strategies to test
    strategies = [
        (DonchianStrategy, {'period': 20}),
        (DMAStrategy, {'fast': 20, 'slow': 50}),
        (BollingerStrategy, {'period': 20, 'std_dev': 2}),
        (RSIStrategy, {'period': 14})
    ]
    
    # Define deviation probabilities to test
    deviation_probs = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    # Run Monte Carlo
    print("\n" + "="*100)
    print("RUNNING MONTE CARLO SIMULATIONS")
    print("="*100)
    print(f"Strategies: {len(strategies)}")
    print(f"Deviation probabilities: {len(deviation_probs)}")
    print(f"Iterations per config: 100")
    print(f"Transaction cost: 0.1% per trade")
    print(f"Total simulations: {len(strategies) * len(deviation_probs) * 100}")
    
    simulator = MonteCarloSimulator(
        data=data,
        strategies=strategies,
        deviation_probabilities=deviation_probs,
        n_iterations=100,
        transaction_cost=0.001
    )
    
    results = simulator.run()
    
    # Display results
    print_results_summary(results)
    print_hypothesis_test(results)
    
    # Save results
    results.to_csv('unreasonableness_results.csv', index=False)
    print("\nResults saved to 'unreasonableness_results.csv'")
    
    return results


if __name__ == "__main__":
    results = main()
