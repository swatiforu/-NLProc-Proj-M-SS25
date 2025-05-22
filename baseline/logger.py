import json
import os
from datetime import datetime


def log_query(data: dict, filepath: str = "logs/pipeline_log.jsonl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data['timestamp'] = datetime.utcnow().isoformat()
    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")
