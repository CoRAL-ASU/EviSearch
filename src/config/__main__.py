"""Show the inference catalog and the current selection.

  python -m src.config           # options and what this run will use
  python -m src.config --check   # also check credentials and local server health
"""
from __future__ import annotations

import argparse
import sys

from src.config.catalog import ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the inference catalog and current selection")
    parser.add_argument("--check", action="store_true", help="Check credentials and local server health")
    args = parser.parse_args(argv)

    try:
        from src.config import config
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    selection = config.SELECTION
    catalog = selection.catalog

    print(f"Preset: {selection.preset}    (available: {', '.join(catalog.presets)}; set EVISEARCH_PRESET)")
    print("\nRoles (EVISEARCH_ROLE_<ROLE>=<model> overrides one)")
    for role in catalog.roles:
        key = selection.roles.get(role)
        where = ""
        if key:
            server = catalog.server_for_model(key)
            where = f"local server {server}" if server else f"endpoint {catalog.models[key].endpoint}"
        print(f"  {role:<15} {key or '-':<24} {where:<32} choices: {', '.join(catalog.models_for_role(role))}")

    print("\nOptions")
    for name, values in catalog.options.items():
        print(f"  {name:<28} {selection.options[name]:<10} ({' | '.join(values)})")

    print(f"\nLocal servers needed (GPU_POOL={selection.gpu_pool})")
    needed = selection.servers_needed()
    for name in needed:
        spec = catalog.servers[name]
        print(
            f"  {name:<16} port {spec.port:<5} GPUs={selection.gpus[name]}  "
            f"tensor_parallel={spec.tensor_parallel}  gpu_memory_utilization={spec.gpu_memory_utilization}"
        )
    if not needed:
        print("  (none)")

    if not args.check:
        return 0

    from src.inference.factory import check_selection

    print("\nChecks")
    problems = check_selection(selection)
    for problem in problems:
        print(f"  ✗ {problem}")
    if not problems:
        print("  ✓ credentials present and local servers reachable")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
