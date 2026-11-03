"""Temporary idle-sleep inhibition for an AC-powered, bounded Kaggle job.

This changes no persistent power setting and does not keep the display on.
Windows releases the execution-state request when this process exits.
"""
import ctypes
import os
from contextlib import contextmanager


@contextmanager
def awake_on_ac():
    enabled=False
    if os.name=='nt':
        class Power(ctypes.Structure):
            _fields_=[('ac',ctypes.c_ubyte),('battery_flag',ctypes.c_ubyte),
                      ('battery_percent',ctypes.c_ubyte),('reserved',ctypes.c_ubyte),
                      ('life',ctypes.c_ulong),('full_life',ctypes.c_ulong)]
        power=Power()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ac==1:
            enabled=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        yield
    finally:
        if enabled:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
