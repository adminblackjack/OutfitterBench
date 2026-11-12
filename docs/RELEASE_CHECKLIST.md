# Before changing the repository from private to public

- Confirm redistribution rights and attribution requirements for third-party
  runtime data. Open-weather provenance is documented; the author still needs
  to confirm the intended redistribution of product listing material.
- Verify `python -m outfitter.public_weather --check`, retain the open source
  snapshots, and review the no-LLM weather-impact results.
- Check that the README identifies historical paper results as using the old
  weather and does not present them as a public-weather leaderboard.
- Add the published arXiv link and accurate BibTeX metadata to the README.
- Check that the result figure matches the final paper and stated protocol.
- Review **all reachable commits**, not just the working tree, for credentials,
  restricted data, private endpoints, personal paths, and unintended files.
- If restricted files were committed during private preparation, deleting them
  is insufficient: prepare another clean initial snapshot before publication.
- Verify installation, no-LLM execution, resume, basic analysis, and tests from a
  fresh clone. Extended evaluation must remain explicitly opt-in.
- Do not publish old development branches, backups, or model trajectories.
- Change visibility manually only after these checks are complete.
- Keep the original repository private and archived. Do not unarchive it or
  import its branches into the new repository as part of public maintenance.
