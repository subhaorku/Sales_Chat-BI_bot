def normalize_region_name(region_name: str | None) -> str | None:
    """
    Converts a region name to its uppercase equivalent.
    Handles None input by returning None.
    """
    if region_name is None:
        return None
    return str(region_name).upper()


def normalize_region_list(region_list: list[str] | None) -> list[str] | None:
    """
    Converts a list of region names to their uppercase equivalents.
    Handles None input or empty list by returning them as is.
    Filters out None values within the list before normalizing.
    """
    if region_list is None:
        return None
    return [normalize_region_name(r) for r in region_list if r is not None]
