from collections.abc import Sequence, Mapping, Set

def create_option(typ: type, default: object, help="", options=[]) -> dict:

    """
    Utility to create specific dictionary structures to keep consistency 
    at the moment to create new options.

    Args:
        typ (type): The type of the option (e.g., str, int, float).
        default (object): The default value for the option, should be a single value.
        help (str): A description of the option.
        options (list): A list of valid values for the option, if applicable.

    Returns:
        dict: A dictionary with the correct structure and previous validations.
    """

    # Check that typ is a valid type object
    if not isinstance(typ, type) or typ is None:
        raise TypeError(f"'typ' must be a type object {typ}' or Unknown type '{typ}' specified.")

    #Check that default is the same type of typ
    if not isinstance(default, typ):
        raise TypeError(f"Default value '{default}' does not match the specified type '{typ}'.")
    
    # Check that default is a single value
    if isinstance(default, (Sequence, Mapping, Set)) and not isinstance(default, (str, bytes)):
        raise TypeError(f"Default value '{default}' must be a single value (a non-collection).")

    #Check that the options match with the type
    if len(options) > 0:
        if not isinstance(options, list):
            raise TypeError("Options must be a list.")
        for opt in options:
            if not isinstance(opt, typ):
                raise TypeError(f"Option '{opt}' does not match the specified type '{typ}'.")


    #Return a dictionary with the correct option format
    return {
        "type": str(typ.__name__),
        "default": default,
        "help": help,
        "options": options
    }
