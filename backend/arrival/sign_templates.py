"""Sign agent templates after human review: python -m arrival.sign_templates arrival/demo/agents.yaml"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from .agents import SIGN_ENV, sign

path = Path(sys.argv[1])
key = os.environ[SIGN_ENV].encode()
doc = yaml.safe_load(path.read_text(encoding="utf-8"))
for t in doc["templates"]:
    t["signature"] = sign(t, key)
path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"signed {len(doc['templates'])} templates")
