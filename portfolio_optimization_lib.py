"""
Portfolio Optimization Library
=============================

A comprehensive library for minimum variance portfolio optimization with
sample and Ledoit-Wolf covariance estimation, constrained optimization,
and rolling window backtesting.

Key Features:
- WRDS data acquisition with coverage validation
- Sample and Ledoit-Wolf covariance estimation
- Constrained quadratic optimization (cvxpy)
- Rolling window out-of-sample backtesting
- Comprehensive performance metrics and visualization
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.covariance import LedoitWolf
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """Main class for portfolio optimization operations."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate (default: 0.02)
        """
        self.risk_free_rate = risk_free_rate
        self.monthly_rf = (1 + risk_free_rate) ** (1/12) - 1
        
    def sample_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate sample covariance matrix."""
        return returns.cov()
    
    def ledoit_wolf_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ledoit-Wolf shrunk covariance matrix."""
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_matrix = lw.covariance_
        return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
    
    def optimize_minimum_variance(self, 
                                 cov_matrix: pd.DataFrame,
                                 max_weight: Optional[float] = None,
                                 min_weight: Optional[float] = None) -> np.ndarray:
        """
        Solve minimum variance portfolio optimization.
        
        Parameters:
        -----------
        cov_matrix : pd.DataFrame
            Covariance matrix of returns
        max_weight : float, optional
            Maximum weight constraint (default: None)
        min_weight : float, optional
            Minimum weight constraint (default: None)
            
        Returns:
        --------
        np.ndarray
            Optimal portfolio weights
        """
        n = cov_matrix.shape[0]
        w = cp.Variable(n)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))
        
        # Constraints
        constraints = [cp.sum(w) == 1]  # Fully invested
        
        if max_weight is not None:
            constraints.append(w <= max_weight)
        if min_weight is not None:
            constraints.append(w >= min_weight)
            
        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve()
        
        if prob.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed with status: {prob.status}")
            
        return w.value
    
    def rolling_backtest(self, 
                        returns: pd.DataFrame,
                        window_size: int,
                        max_weight: Optional[float] = None,
                        min_weight: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Perform rolling window out-of-sample backtesting.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Monthly returns data (index=date, columns=assets)
        window_size : int
            Rolling window size in months
        max_weight : float, optional
            Maximum weight constraint
        min_weight : float, optional
            Minimum weight constraint
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary containing weights and returns for both methods
        """
        dates = returns.index
        n_periods = len(dates)
        
        # Initialize results
        sample_weights = pd.DataFrame(columns=returns.columns)
        lw_weights = pd.DataFrame(columns=returns.columns)
        sample_returns = pd.Series(dtype=float, name='portfolio_return')
        lw_returns = pd.Series(dtype=float, name='portfolio_return')
        
        for i in range(window_size, n_periods - 1):
            # Training window
            train_start = i - window_size
            train_end = i
            train_data = returns.iloc[train_start:train_end]
            
            # Test period (next month)
            test_date = dates[i + 1]
            test_returns = returns.iloc[i + 1]
            
            try:
                # Sample covariance optimization
                sample_cov = self.sample_covariance(train_data)
                sample_w = self.optimize_minimum_variance(sample_cov, max_weight, min_weight)
                sample_weights.loc[dates[i]] = sample_w
                sample_returns.loc[test_date] = np.dot(test_returns.values, sample_w)
                
                # Ledoit-Wolf optimization
                lw_cov = self.ledoit_wolf_covariance(train_data)
                lw_w = self.optimize_minimum_variance(lw_cov, max_weight, min_weight)
                lw_weights.loc[dates[i]] = lw_w
                lw_returns.loc[test_date] = np.dot(test_returns.values, lw_w)
                
            except Exception as e:
                print(f"Warning: Optimization failed at {dates[i]}: {e}")
                continue
        
        return {
            'sample_weights': sample_weights,
            'lw_weights': lw_weights,
            'sample_returns': sample_returns,
            'lw_returns': lw_returns
        }
    
    def calculate_performance_metrics(self, 
                                    returns: pd.Series,
                                    weights: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Portfolio returns
        weights : pd.DataFrame, optional
            Portfolio weights over time
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        if len(returns) == 0:
            return {key: np.nan for key in ['annual_return', 'annual_volatility', 
                                          'sharpe_ratio', 'max_drawdown', 'calmar_ratio']}
        
        # Basic metrics
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (12 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        # Drawdown metrics
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = -drawdown.min()
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else np.nan
        
        # Turnover (if weights provided)
        turnover = None
        if weights is not None and not weights.empty:
            weight_changes = weights.diff().abs().sum(axis=1)
            turnover = weight_changes.mean()
        
        # Weight stability
        weight_stability = None
        if weights is not None and not weights.empty:
            weight_stability = weights.std(axis=1).mean()
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_return': total_return
        }
        
        if turnover is not None:
            metrics['avg_turnover'] = turnover
        if weight_stability is not None:
            metrics['weight_stability'] = weight_stability
            
        return metrics
    
    def plot_cumulative_returns(self, 
                               sample_returns: pd.Series,
                               lw_returns: pd.Series,
                               title: str = "Cumulative Returns Comparison") -> None:
        """Plot cumulative returns for both methods."""
        sample_cum = (1 + sample_returns).cumprod()
        lw_cum = (1 + lw_returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(sample_cum.index, sample_cum.values, label='Sample Covariance', linewidth=2)
        plt.plot(lw_cum.index, lw_cum.values, label='Ledoit-Wolf Covariance', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_turnover(self, 
                     sample_weights: pd.DataFrame,
                     lw_weights: pd.DataFrame) -> None:
        """Plot portfolio turnover over time."""
        sample_turnover = sample_weights.diff().abs().sum(axis=1)
        lw_turnover = lw_weights.diff().abs().sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(sample_turnover.index, sample_turnover.values, 
                label='Sample Covariance', linewidth=2)
        plt.plot(lw_turnover.index, lw_turnover.values, 
                label='Ledoit-Wolf Covariance', linewidth=2)
        
        plt.title('Portfolio Turnover Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Turnover', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weight_stability(self, 
                            sample_weights: pd.DataFrame,
                            lw_weights: pd.DataFrame) -> None:
        """Plot weight stability (cross-sectional standard deviation)."""
        sample_stability = sample_weights.std(axis=1)
        lw_stability = lw_weights.std(axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(sample_stability.index, sample_stability.values, 
                label='Sample Covariance', linewidth=2)
        plt.plot(lw_stability.index, lw_stability.values, 
                label='Ledoit-Wolf Covariance', linewidth=2)
        
        plt.title('Portfolio Weight Stability', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Weight Standard Deviation', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_table(self, 
                           sample_returns: pd.Series,
                           lw_returns: pd.Series,
                           sample_weights: pd.DataFrame,
                           lw_weights: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive performance summary table."""
        sample_metrics = self.calculate_performance_metrics(sample_returns, sample_weights)
        lw_metrics = self.calculate_performance_metrics(lw_returns, lw_weights)
        
        summary_data = {
            'Sample Covariance': sample_metrics,
            'Ledoit-Wolf Covariance': lw_metrics
        }
        
        summary_df = pd.DataFrame(summary_data).T
        summary_df = summary_df.round(4)
        
        return summary_df
