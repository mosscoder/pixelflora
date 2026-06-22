"""Publish the assembled dataset to the Hugging Face Hub, PRIVATE by default.

Each species is pushed as its own configuration of the same repository, so a new
species can be added later as a new configuration without disturbing the existing
ones. Only pushes when the request opts in (``publish.push = true``) and a token
is present. Otherwise it is a dry run: the artifacts are written locally and we
report what would be pushed.

We let ``push_to_hub`` own the repository README, because that is where the
``datasets`` library records the configuration list that the dataset viewer
reads. The provenance summary is uploaded alongside as PROVENANCE.md so it does
not overwrite that list. The per image license and attribution always live in the
dataset rows themselves, so the data stays self describing.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from datasets import DatasetDict

from .config import Config
from .request import RequestSpec


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "pixelflora-dataset"


def resolve_repo_id(spec: RequestSpec, config: Config) -> str | None:
    repo = spec.publish.repo_id
    if repo and "/" in repo:
        return repo
    if repo:
        base = repo
    elif spec.dataset.name:
        base = _slug(spec.dataset.name)
    elif len(spec.species) == 1:
        base = _slug(f"{spec.species[0].genus}-{spec.species[0].species}")
    else:
        base = "botanical-multispecies"
    owner = config.publish_owner
    return f"{owner}/{base}" if owner else base


def publish(configs: "OrderedDict[str, DatasetDict]", out_dir: str,
            spec: RequestSpec, config: Config) -> dict:
    repo_id = resolve_repo_id(spec, config)
    private = spec.publish.private if spec.publish.private is not None else config.publish_private
    names = list(configs.keys())

    if not spec.publish.push:
        return {"pushed": False, "repo_id": repo_id, "private": private,
                "configs": names,
                "reason": "push not requested (dry run); artifacts written locally"}
    if not config.hf_token:
        return {"pushed": False, "repo_id": repo_id, "private": private,
                "configs": names,
                "reason": "no HF token in env (HF_TOKEN); skipped push"}

    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", private=private,
                token=config.hf_token, exist_ok=True)
    # one configuration per species; existing configurations are left untouched
    for name, dd in configs.items():
        dd.push_to_hub(repo_id, config_name=name, private=private, token=config.hf_token)
    # provenance summary alongside the datasets-managed README (PROVENANCE.md so it
    # does not overwrite the configuration list the viewer relies on)
    for local_name, repo_name in (("README.md", "PROVENANCE.md"),
                                  ("CITATION.cff", "CITATION.cff"),
                                  ("bibliography.bib", "bibliography.bib"),
                                  ("license_report.json", "license_report.json")):
        fp = Path(out_dir) / local_name
        if fp.exists():
            upload_file(path_or_fileobj=str(fp), path_in_repo=repo_name,
                        repo_id=repo_id, repo_type="dataset", token=config.hf_token)
    return {"pushed": True, "repo_id": repo_id, "private": private, "configs": names}
