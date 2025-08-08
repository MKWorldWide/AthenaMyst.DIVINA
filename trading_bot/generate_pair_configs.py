#!/usr/bin/env python3
"""
Generate individual OANDA config files for each trading pair from the main config.
"""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load the main config file
    main_config = Path('.env.oanda.multi')
    if not main_config.exists():
        print(f"Error: {main_config} not found in current directory")
        return
    
    # Load environment variables
    load_dotenv(main_config)
    
    # Get trading pairs
    trading_pairs = os.getenv('TRADING_PAIRS', '').split(',')
    if not trading_pairs or not trading_pairs[0]:
        print("Error: No trading pairs found in TRADING_PAIRS")
        return
    
    # Create backup of main config
    backup_config = main_config.with_suffix('.backup')
    if not backup_config.exists():
        shutil.copy(main_config, backup_config)
        print(f"Created backup at {backup_config}")
    
    # Create individual configs
    created = 0
    for pair in trading_pairs:
        pair = pair.strip()
        if not pair:
            continue
            
        # Create pair-specific config
        pair_config = Path(f'.env.oanda.{pair}')
        if pair_config.exists():
            print(f"Skipping existing: {pair_config}")
            continue
            
        # Copy main config to pair config
        shutil.copy(main_config, pair_config)
        print(f"Created: {pair_config}")
        created += 1
    
    print(f"\nGenerated {created} pair configs from {main_config}")
    print("\nNext steps:")
    print("1. Review and edit the generated configs if needed")
    print("2. Run: python oanda_multi_scalper.py")

if __name__ == "__main__":
    main()
