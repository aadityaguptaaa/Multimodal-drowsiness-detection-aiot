#!/usr/bin/env python3
"""
Launcher script referencing the root bridge.py
"""
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_bridge = os.path.join(root_dir, "bridge.py")

if __name__ == "__main__":
    if os.path.exists(root_bridge):
        # Execute root bridge.py in same Python process
        with open(root_bridge, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, {"__name__": "__main__", "__file__": root_bridge})
    else:
        print(f"Error: Could not find {root_bridge}")
        sys.exit(1)
