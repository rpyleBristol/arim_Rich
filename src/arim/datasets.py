"""
Example datasets

Usage::

    fname = arim.datasets.EXAMPLES.fetch("contact_notch_aluminium.mat")


"""

from typing import Dict

import pooch

EXAMPLES = pooch.create(
    path=pooch.os_cache("arim"),
    base_url="https://github.com/rpyleBristol/arim_Rich/raw/fb8ab4183f068e2e95e1a3a8e389042ad0e6b309/examples/example-datasets/",
    version=None,
    version_dev=None,
    registry={
        "contact_notch_aluminium.mat": "sha256:c58d25ccaf3081348c392d98a45a5e28809c80c900782a9b60f62d1a7ef2c76f",
        "immersion_notch_aluminium.mat": "sha256:ae125a6c461a10ded54ca19e223c4d39fe857be8c12fda6f97b61de25a52a08e",
        "contact_sdh_aluminium_nlos.mat": "sha256:f66df7ddfc236222b3fdc35a6cd4247661474b03d3918fdb4c1ec63752e6f0c0",
        "PWC_immersion_notch_stainless_steel.mat": "sha256:6aa5f205bf0d9736cd7b20cb97bc02d7244282f8d77b3cf60ef2698cc18e3ee0",
    },
)

# Dictionary of all datasets
DATASETS: Dict[str, pooch.Pooch] = {"examples": EXAMPLES}
