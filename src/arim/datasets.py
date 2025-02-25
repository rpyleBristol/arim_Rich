"""
Example datasets

Usage::

    fname = arim.datasets.EXAMPLES.fetch("contact_notch_aluminium.mat")


"""

from typing import Dict

import pooch

EXAMPLES = pooch.create(
    path=pooch.os_cache("arim"),
    base_url="https://github.com/rpyleBristol/arim_Rich/raw/b8e0febd986abedaf29ec0d13f9aeaf066b90900/examples/example-datasets/",
    version=None,
    version_dev=None,
    registry={
        "contact_notch_aluminium.mat": "sha256:c58d25ccaf3081348c392d98a45a5e28809c80c900782a9b60f62d1a7ef2c76f",
        "immersion_notch_aluminium.mat": "sha256:ae125a6c461a10ded54ca19e223c4d39fe857be8c12fda6f97b61de25a52a08e",
        "contact_sdh_aluminium_nlos.mat": "sha256:f66df7ddfc236222b3fdc35a6cd4247661474b03d3918fdb4c1ec63752e6f0c0",
        "PWC_immersion_notch_stainless_steel.mat": "sha256:f0ab03abe482062268bc67e8a182ec594c4bfc0bf0bc6d9c3f2ee97066b1ea52",
    },
)

# Dictionary of all datasets
DATASETS: Dict[str, pooch.Pooch] = {"examples": EXAMPLES}
