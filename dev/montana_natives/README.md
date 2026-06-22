# Dev example — Montana native forbs (3 species)

A worked multi-class case: three Montana grassland natives in one request →
one labelled dataset.

| class (`label`) | common name |
|---|---|
| *Lupinus sericeus* | silky lupine |
| *Gaillardia aristata* | blanketflower |
| *Achillea millefolium* | common yarrow |

```bash
mamba run -n pixelflora pixelflora run dev/montana_natives/request.toml
```

Produces `out/` with a balanced 3-class dataset (up to 100 images per class,
files nested under `out/images/<Genus_species>/`), a random split applied
**within each class** (so every class appears in train and test), full per-image
attribution, and the provenance card with a per-class breakdown.
What it demonstrates on **real** data:

- **Multiple species in one TOML** via `[[species]]`; each row carries a `label`.
- **Per-class image cap + balance** (`max_images` is per species).
- **Attribution end to end** (source URL, license, rights-holder, creator, link).
- **iNat-native signals** — phenology, coordinate obscuring, captive exclusion,
  identification-agreement counts.
- **1024 px cap** (`max_dimension`) keeping the dataset light.

Swap the split to `geographic` or `temporal` for spatial/temporal generalization
tests (give each class enough images first — thin classes can't fill both splits
under blocking).
