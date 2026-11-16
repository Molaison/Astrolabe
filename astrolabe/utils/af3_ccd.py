"""Construct the pinned AF3 CCD without modifying its process-wide base data."""


def create_isolated_ccd(chemical_components, ccd_pickle_path=None, user_ccd=None):
    """Return a native CCD with a private dictionary for custom components.

    The pinned AF3 constructor overlays ``user_ccd`` into its cached base dict.
    Build the same native class and parse the same CIF payload, but apply that
    overlay to a shallow copy. Unchanged entries keep their stable identities,
    which the native reference-conformer cache uses. Callers that featurise
    with a custom CCD must retain it or clear identity-keyed reference entries
    before releasing it; AF3CcdCache owns that lifetime in prediction.
    """
    ccd = (chemical_components.Ccd() if ccd_pickle_path is None
           else chemical_components.Ccd(ccd_pickle_path=ccd_pickle_path))
    if user_ccd is not None:
        if not user_ccd:
            raise ValueError("User CCD cannot be an empty string.")
        entries = chemical_components.cif_dict.parse_multi_data_cif(user_ccd)
        ccd._dict = dict(ccd.items())
        ccd._dict.update({
            name: {key: tuple(value) for key, value in entry.items()}
            for name, entry in entries.items()
        })
    return ccd
