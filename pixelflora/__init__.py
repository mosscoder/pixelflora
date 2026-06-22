"""pixelflora — attributed botanical image datasets for model training.

Pipeline: resolve -> harvest -> filter -> download -> assemble -> split -> publish.
Sources implement a common interface (``pixelflora.sources.base.Source``) so new
providers (GBIF, iNaturalist, ...) plug in without touching the rest of the pipeline.
"""

__version__ = "0.1.0"
