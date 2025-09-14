#!/usr/bin/env python3
"""
Special version of scalar bounds calculator for Unichain using only last 3 months.
Produces the same visualization style as Base/Arbitrum.
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import the main scalar bounds calculator
sys.path.insert(0, str(Path(__file__).parent))
from scalar_bounds_calculator import ScalarBoundsCalculator, ScalarBoundsResult, main


class UnichainScalarBoundsCalculator(ScalarBoundsCalculator):
    """Modified calculator that uses only recent 3 months for Unichain."""
    
    def __init__(self, chain: str, lookback_months: int = 3, period_days: int = 21):
        # Force 3-month lookback for Unichain
        super().__init__(chain, lookback_months=3, period_days=period_days)
        
    def calculate_log_returns(self, period_days: int = None) -> np.ndarray:
        """
        Override to use only last 3 months of data.
        """
        if period_days is None:
            period_days = self.period_days
            
        if not self.tvl_data:
            self.fetch_tvl_history()
            
        timestamps = sorted(self.tvl_data.keys())
        
        # Use only last 3 months
        end_date = datetime.fromtimestamp(timestamps[-1])
        start_date = end_date - timedelta(days=90)  # 3 months = 90 days
        start_timestamp = int(start_date.timestamp())
        
        filtered_timestamps = [t for t in timestamps if t >= start_timestamp]
        
        print(f"Using only last 3 months: {len(filtered_timestamps)} days of data")
        
        # Calculate log-returns
        log_returns = []
        period_seconds = period_days * 86400
        
        for t in filtered_timestamps:
            target_timestamp = t + period_seconds
            future_timestamps = [ts for ts in filtered_timestamps if ts >= target_timestamp]
            
            if not future_timestamps:
                continue
                
            future_timestamp = future_timestamps[0]
            start_tvl = self.tvl_data[t]
            end_tvl = self.tvl_data[future_timestamp]
            
            if start_tvl > 0 and end_tvl > 0:
                # Log-return: log(end/start)
                log_return = np.log(end_tvl / start_tvl)
                log_returns.append(log_return)
                
        return np.array(log_returns)


def main_unichain(period_days: int = 21):
    """Main execution for Unichain with 3-month data."""
    
    # Initialize calculator with 3-month lookback
    calculator = UnichainScalarBoundsCalculator(chain="Unichain", lookback_months=3, period_days=period_days)
    
    # Calculate bounds
    result = calculator.calculate_bounds(confidence_level=0.99)
    
    # Print summary
    calculator.print_summary(result)
    
    # Save results
    results_dir = Path("portfolio_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save with special name to indicate 3-month data
    from dataclasses import asdict
    import json
    
    results_file = results_dir / f"unichain_scalar_bounds_{period_days}d_3m.json"
    with open(results_file, 'w') as f:
        result_dict = asdict(result)
        result_dict['analysis_date'] = datetime.now().isoformat()
        result_dict['data_window'] = '3_months'
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Visualize with same style as Base/Arbitrum
    viz_file = results_dir / f"unichain_scalar_bounds_{period_days}d_3m_viz.png"
    calculator.visualize_comparison(result, save_path=str(viz_file))
    
    return result


if __name__ == "__main__":
    period_days = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    result = main_unichain(period_days)