#!/usr/bin/env python3
"""
Helper script to setup plugin allowlist.

Run this to create ~/.ocr-vibe/allowed_plugins.json with hashes of trusted plugins.
Usage:
    python scripts/setup_plugin_allowlist.py
"""
import hashlib
import json
import sys
from pathlib import Path


def get_plugin_hash(plugin_path: Path) -> str:
    """Calculate SHA256 hash of a plugin file."""
    with open(plugin_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def setup_allowlist(engines_dir: Path | None = None):
    """
    Scan engines directory and create allowlist.

    Args:
        engines_dir: Path to engines directory. Defaults to ~/.ocr-vibe/engines/
    """
    if engines_dir is None:
        engines_dir = Path.home() / '.ocr-vibe' / 'engines'

    if not engines_dir.is_dir():
        print(f"Engines directory not found: {engines_dir}")
        print("Creating empty allowlist...")
        allowlist = {'allowed': [], 'hashes': {}}
    else:
        allowed = []
        hashes = {}
        for plugin_file in sorted(engines_dir.glob('*.py')):
            if plugin_file.name.startswith('_'):
                continue
            try:
                h = get_plugin_hash(plugin_file)
                plugin_name = plugin_file.stem
                allowed.append(plugin_name)
                hashes[plugin_name] = h
                print(f"  + {plugin_name}: {h[:16]}...")
            except IOError as e:
                print(f"  ! Error reading {plugin_file.name}: {e}")

        allowlist = {'allowed': allowed, 'hashes': hashes}

    output_path = Path.home() / '.ocr-vibe' / 'allowed_plugins.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(allowlist, f, indent=2)

    print(f"\nAllowlist saved to: {output_path}")
    print(f"Total plugins: {len(allowed)}")


if __name__ == '__main__':
    setup_allowlist()
