"""pixelflora CLI.

    pixelflora run     <request.toml>     # full pipeline (default)
    pixelflora resolve <request.toml>     # just resolve the taxon on each source
    pixelflora harvest <request.toml>     # harvest + filter, write manifests (no downloads)

Most work is done via `run`; the others are for inspecting intermediate stages.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .request import RequestSpec


def _cmd_run(args):
    from .pipeline import run
    run(args.request, Config.load(args.config))


def _cmd_resolve(args):
    from .http import PoliteClient
    from .sources import get_source
    config = Config.load(args.config)
    spec = RequestSpec.from_toml(args.request)
    client = PoliteClient(user_agent=config.user_agent, timeout_s=config.timeout_s,
                          max_retries=config.max_retries, rate_limit_s=config.rate_limit_s,
                          cache_dir=config.cache_dir)
    for sp in spec.species:
        for name in spec.sources:
            src = get_source(name, config, client)
            t = src.resolve_taxon(sp.genus, sp.species, taxon_key=sp.taxon_key)
            print(json.dumps(t.model_dump(), indent=2))


def _cmd_harvest(args):
    from pathlib import Path
    from .http import PoliteClient
    from .filters import apply_filters
    from .sources import get_source
    config = Config.load(args.config)
    spec = RequestSpec.from_toml(args.request)
    client = PoliteClient(user_agent=config.user_agent, timeout_s=config.timeout_s,
                          max_retries=config.max_retries, rate_limit_s=config.rate_limit_s,
                          cache_dir=config.cache_dir)
    records = []
    for sp in spec.species:
        for name in spec.sources:
            src = get_source(name, config, client)
            t = src.resolve_taxon(sp.genus, sp.species, taxon_key=sp.taxon_key)
            got = 0
            for rec in src.harvest(spec, t):
                rec.label = RequestSpec.label_of(sp)
                records.append(rec)
                got += len(rec.images)
                if got >= spec.media.max_images * 3:
                    break
    kept, reasons = apply_filters(records, spec.filters)
    out = Path(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "manifest.filtered.jsonl").open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r.model_dump(exclude={"raw"}), default=str) + "\n")
    print(f"harvested {len(records)} -> kept {len(kept)} (rejections={dict(reasons)})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pixelflora", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to config.toml (default: auto-discover)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, fn in [("run", _cmd_run), ("resolve", _cmd_resolve), ("harvest", _cmd_harvest)]:
        sp = sub.add_parser(cmd)
        sp.add_argument("request", help="path to a request .toml")
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
