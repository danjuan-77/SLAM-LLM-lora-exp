

def get_first_existing_value(data_dict, keys, default=None):
    """
    Returns the value for the first key found in the dictionary from a list of possible keys.
    Skips keys that are missing or whose values are None.

    Args:
        data_dict (dict): The dictionary to search.
        keys (List[str]): A list of candidate keys, in priority order.
        default: Value to return if none of the keys exist or all have None values.

    Returns:
        Any: The value corresponding to the first existing key, or the default value.
    """
    for key in keys:
        if key in data_dict and data_dict[key] is not None:
            return data_dict[key]
    return default
