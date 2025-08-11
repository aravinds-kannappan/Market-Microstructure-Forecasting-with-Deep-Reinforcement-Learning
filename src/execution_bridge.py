"""
execution_bridge.py
===================
Python wrapper around a compiled C++ execution engine (libexecution_engine.so).
Compile with:
  g++ -O3 -std=c++17 -shared -fPIC src/execution_engine.cpp -o src/libexecution_engine.so
"""
import ctypes, os

LIB_PATH = os.path.join(os.path.dirname(__file__), "libexecution_engine.so")

class ExecutionBridge:
    def __init__(self):
        if not os.path.exists(LIB_PATH):
            raise FileNotFoundError(f"Shared lib not found at {LIB_PATH}. Compile execution_engine.cpp first.")
        self.lib = ctypes.CDLL(LIB_PATH)
        # Example: set arg/return types when you expose functions in C++

    # Placeholder methods
    def place_order(self, side: str, price: float, qty: int):
        pass

    def cancel_order(self, order_id: int):
        pass
