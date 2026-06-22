"""Publish the assembled dataset to the Hugging Face Hub — PRIVATE by default.

Only pushes when the request explicitly opts in (``publish.push = true``) and a
token is present. Otherwise it's a dry run: the card and citation artifacts are
written locally and we report what *would* be pushed.
"""
from __future__ import annotations

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


def publish(dd: DatasetDict, out_dir: str, spec: RequestSpec, config: Config) -> dict:
    repo_id = resolve_repo_id(spec, config)
    private = spec.publish.private if spec.publish.private is not None else config.publish_private
    do_push = spec.publish.push and config.publish_push is not False  # request must opt in

    if not (do_push and spec.publish.push):
        return {"pushed": False, "repo_id": repo_id, "private": private,
                "reason": "push not requested (dry run); artifacts written locally"}
    if not config.hf_token:
        return {"pushed": False, "repo_id": repo_id, "private": private,
                "reason": "no HF token in env (HF_TOKEN); skipped push"}

    from huggingface_hub import HfApi, create_repo, upload_file

    create_repo(repo_id, repo_type="dataset", private=private,
                token=config.hf_token, exist_ok=True)
    dd.push_to_hub(repo_id, private=private, token=config.hf_token)
    api = HfApi(token=config.hf_token)
    for fname in ("README.md", "CITATION.cff", "bibliography.bib", "license_report.json"):
        fp = Path(out_dir) / fname
        if fp.exists():
            upload_file(path_or_fileobj=str(fp), path_in_repo=fname,
                        repo_id=repo_id, repo_type="dataset", token=config.hf_token)
    return {"pushed": True, "repo_id": repo_id, "private": private}
