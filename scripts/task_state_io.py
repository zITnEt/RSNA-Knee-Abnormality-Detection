"""Atomic progress updates tolerant of short Windows reader locks."""
import json
import os
import time


def atomic_json_write(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.05*(attempt+1),.5))
