<p align="center">
  <img src="assets/pixelflora_logo.jpg" alt="pixelflora" width="480">
</p>

# pixelflora

pixelflora gathers photographs of wild plants from iNaturalist, the citizen
science platform where naturalists record what they find in the field. You name
a species, and pixelflora downloads its images together with the full account of
each photograph: who made it, under what license, where and when the plant was
seen, and how it was identified. It then arranges those images and their records
into a dataset in the Hugging Face format, ready to train an image recognition
model.

The dataset is meant to stay private. The photographs belong to the people who
made them and are used here under their original licenses to train a model, not
to be republished. pixelflora keeps every dataset private by default and writes
down each license rather than discarding images, so that the record of where the
data came from is complete enough to cite in a publication.

## How it works

```
resolve  →  harvest  →  filter  →  download  →  assemble  →  split  →  publish
```

pixelflora first resolves the species name to its iNaturalist taxon. It then
harvests the records, meaning the written information only and not yet the
images. It applies the filters you asked for, downloads the image files, arranges
them into a dataset, divides them into a training set and a test set, and can
publish the result to a private repository on the Hugging Face Hub. The written
records are kept apart from the image files, so you can refilter, redivide, or
republish a dataset without downloading anything again, and so that every run can
be reproduced exactly.

## Installing

pixelflora runs in a conda environment. With mamba installed:

```bash
mamba create -n pixelflora -c conda-forge python=3.12 pip
mamba run -n pixelflora pip install -e .
```

## Configuring

Settings that are not secret live in a file named `config.toml`, written in the
same TOML format as the requests below. The one secret, needed only when you
publish, is read from the environment:

* `HF_TOKEN`, your Hugging Face access token, used only to push private datasets
  to the Hub.

## Running it

Each task is described by a single TOML file that names the species and any
filters, image options, division settings, and publishing settings. The commands:

```bash
pixelflora run     dev/montana_natives/request.toml    # the whole sequence
pixelflora resolve dev/montana_natives/request.toml    # only resolve the taxon
pixelflora harvest dev/montana_natives/request.toml    # harvest and filter, no downloads
```

A run writes everything to the directory you set under `[output]`:

* `images/`, the downloaded photographs, with identical files removed by sha256
  checksum and grouped into one folder per species.
* `dataset/`, the dataset itself in Hugging Face format (Arrow and Parquet, with
  the image data included).
* `manifest.raw.jsonl`, every record that was harvested, with its full account of
  origin.
* `manifest.filtered.jsonl`, the records that passed your filters.
* `manifest.images.jsonl`, the same records after downloading, now carrying each
  file's checksum, dimensions, and path.
* `metadata.csv`, a flat table with one row for each image, for easy reading.
* `README.md`, a written summary of the dataset's origins, including the
  breakdown of licenses and of who contributed.
* `bibliography.bib`, `CITATION.cff`, `license_report.json`, `rejections.json`,
  and `summary.json`.

## Several species in one request

To build a dataset that tells several species apart, list each one as a
`[[species]]` entry, which is a TOML array of tables. A single `[species]` entry
still works when you want only one. Every image is given a `label` of the form
`Genus species`. The image limit `max_images` applies to each species on its own,
so the classes stay balanced, and the division into training and test sets is
done inside each species, so that every species appears in both sets no matter
which method of division you choose.

```toml
[[species]]
genus = "Lupinus"
species = "sericeus"

[[species]]
genus = "Lupinus"
species = "argenteus"
```

See `dev/montana_natives/` for a worked example of three species.

## What is recorded for each image

* Origin: the observer, the iNaturalist observation it came from, the license,
  the rights holder, and a finished attribution line.
* Taxonomy: the scientific name, the common name, and the broad group (for
  plants, Plantae).
* Place: latitude and longitude, the stated uncertainty, a marker noting when
  iNaturalist has obscured the location to protect a sensitive species, and the
  locality description.
* Time: the date observed, the year, and the day of the year, which is useful for
  phenology.
* Botanical traits: the flowering or fruiting state taken from the iNaturalist
  phenology annotation and stored as `reproductive_condition`, and whether the
  plant was wild or cultivated, stored as `establishment_means`.
* Curation quality: whether the observation reached research grade, and how many
  identifiers agreed or disagreed.

The full list of fields is in `pixelflora/schema.py`.

## Filtering

Filters are optional and chosen in the request. They include whether a record has
coordinates, whether it must be research grade (which is on by default), whether
to drop cultivated plants and keep only wild ones (`exclude_captive`), a range of
years, a largest allowed coordinate uncertainty, a flowering state
(`reproductive_condition`, which iNaturalist can filter for you), and the license.
Filtering by license is off by default, because the datasets are private.

## Dividing the data into training and test sets

pixelflora offers three ways to divide the images, what machine learning calls a
split:

* `geographic`, which uses spatial blocking. It sorts the records into cells on a
  grid of latitude and longitude and assigns whole cells to one set or the other,
  so that photographs taken at the same place never fall on both sides. This
  guards against a model that has only memorized locations.
* `temporal`, which holds back the most recent records, or everything from a
  chosen year onward, to test how well the model carries across time.
* `random`, a shuffled division with a fixed seed, balanced if you wish across
  the values of any field.

## Adding other data sources

Each data source is a single Python class under `pixelflora/sources/`. At present
the one source is iNaturalist (`sources/inaturalist.py`). The arrangement is kept
general, so that another source could be added later by writing one new class,
without changing the rest of the program.

## Licensing and responsible use

The pixelflora code is released under the MIT license (see `LICENSE`). The
photographs and records it gathers stay under their own licenses and under
iNaturalist's [Terms of Use](https://www.inaturalist.org/pages/terms). pixelflora
writes down the license and attribution for every item and keeps datasets private
by default. By default it makes no more than about one request each second,
following iNaturalist's guidance. If you pass anything along to others, honor each
item's license and attribution.
