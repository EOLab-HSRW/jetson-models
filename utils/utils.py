def create_option(type_, default, help="", options=None):
    """
    Utility to create specific dictionary structures to keep consistency 
    at the moment to create new options.

    Args:
        type_ (str): The type of the option (e.g., "string", "integer").
        default: The default value for the option.
        help (str): A description of the option.
        options (list): A list of valid values for the option, if applicable.

    Returns:
        dict: A dictionary with the correct structure and previous validations.
    """

    # Type mapping
    type_mapping = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool
    }

    # Check that the default value matches the specified type
    expected_type = type_mapping[type_]
    if expected_type is None:
        raise ValueError(f"Unknown type '{type_}' specified.")

    if not isinstance(default, expected_type):
        raise TypeError(f"Default value '{default}' does not match the specified type '{type_}'.")

    #Check that the options match with the default type
    if options is not None:
        if not isinstance(options, list):
            raise TypeError("Options must be a list.")
        for opt in options:
            if not isinstance(opt, type(default)):
                raise TypeError(f"Option '{opt}' does not match the specified type '{type_}'.")

    #Return a dictionary with the correct option format
    return {
        "type": type_,
        "default": default,
        "help": help,
        "options": options if options else []
    }
