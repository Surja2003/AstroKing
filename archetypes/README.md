# Archetypes (starter pack)

Drop a handful of palm photos into each subfolder. The folder name becomes the archetype `label` returned by the backend.

Suggested minimum:
- 5–20 images per archetype (more is better)
- Keep capture conditions consistent (lighting, distance, background)

Example structure:

- `archetypes/`
  - `The Strategist/`
  - `The Empath/`
  - `The Builder/`

Optional metadata lives in `archetypes_meta.json`.

## Safety note

This is a prototype “retrieval” system: it finds the closest archetype based on visual similarity in embedding space.
Treat outputs as entertainment/insight prompts, not factual claims about a person.
