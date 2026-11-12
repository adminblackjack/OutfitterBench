"""Export the configured actor inputs for review, without creating an LLM client."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from .config import OutfitterConfig
from .data import OutfitterDataset
from .environment import OutfitterEnvironment
from .prompts import system_prompt
from .runner import FIRST_ONLY_PROTOCOL, READ_ONLY_BATCH_PROTOCOL
from .tool_executor import ToolExecutor


def export_actor_interface(config: OutfitterConfig, output_dir: Path, stem: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        raise ValueError("stem must use letters, digits, underscore or dash")
    env = OutfitterEnvironment(config, OutfitterDataset.load(config.data_dir))
    tools = ToolExecutor(env)
    if tools.business_rules_snapshot is None:
        raise ValueError("this exporter requires the public business manual interface")
    prompt = system_prompt(
        env.public_operating_rules(),
        tool_call_protocol=READ_ONLY_BATCH_PROTOCOL if config.parallel_tool_calls else FIRST_ONLY_PROTOCOL,
        read_only_tools=tools.read_only_names(),
        max_read_only_batch_size=config.max_read_only_batch_size,
    )
    manual = tools.business_rules_snapshot
    files = {
        "system_prompt.md": prompt,
        "business_rules.json": json.dumps(manual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "business_rules.md": "# Public business rules\n\n" + "\n\n".join(
            f"## {topic}: {chapter['title']}\n\n" + "\n\n".join(chapter["rules"])
            for topic, chapter in manual.items()
        ) + "\n",
        "tool_definitions.json": json.dumps(tools.definitions(), ensure_ascii=False, indent=2) + "\n",
    }
    manifest = {
        "environment_protocol": env.dataset.metadata["environment_protocol"],
        "tool_interface_version": tools.tool_interface_version,
        "assortment_mode": config.assortment_mode,
        "system_prompt_characters": len(prompt),
        "note": "System prompt is exact; handbook is queried, not injected in full. Tools are sent each response. No model was called.",
        "files": {f"{stem}.{suffix}": hashlib.sha256(text.encode("utf-8")).hexdigest()
                  for suffix, text in files.items()},
    }
    files["export_manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_dir = Path(output_dir)
    if any((output_dir / f"{stem}.{suffix}").exists() for suffix in files):
        raise ValueError("export target already exists; choose a new stem")
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, text in files.items():
        with (output_dir / f"{stem}.{suffix}").open("x", encoding="utf-8") as stream:
            stream.write(text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    args = parser.parse_args()
    print(json.dumps(export_actor_interface(
        OutfitterConfig.from_yaml(args.config), args.output_dir, args.stem), indent=2))


if __name__ == "__main__":
    main()
