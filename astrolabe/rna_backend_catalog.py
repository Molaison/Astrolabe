"""RNA backend capability metadata; standard-library imports only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RnaBackend:
    name: str
    class_name: str
    source: str
    revision: str
    experimental_multichain: bool = False
    multichain_limitation: str = ""


RNA_BACKENDS = {
    spec.name: spec
    for spec in (
        RnaBackend("drfold2", "DRfold2Backend", "leeyang/DRfold2",
                   "3667d6ac44c5b14cff94c36e0371224acb2cdae5",
                   multichain_limitation=(
                       "Both RCLM and structure positions assume one polymer; "
                       "PotentialFold adds O3'-P bonds and bonded clash exclusions "
                       "across every consecutive residue. Chain-aware refinement "
                       "and homomer initialization are not implemented."
                   )),
        RnaBackend("numonomer", "NUMonomerBackend", "yunda-si/NUMonomer",
                   "af38434e7ee3f82c6948841cce1ca4d65840742a", True),
        RnaBackend("zfold", "ZfoldBackend", "heqin-zhu/structRFM",
                   "f0c1add19f8dfc786f16bc4abb7aa775dbadddbf",
                   multichain_limitation=(
                       "structRFM/crop and geometry positions lack chain boundaries; "
                       "the Rosetta path builds one covalently connected polymer, "
                       "not disconnected chains with movable interchain jumps."
                   )),
        RnaBackend("rnabpflow", "RNAbpFlowBackend", "Bhattacharya-Lab/RNAbpFlow",
                   "e8b1c077f7951d4d6ebe00c851d22c37777dfc9f", True),
    )
}

