import os
import shutil
import numpy as np
import jetson_utils
from typing import Dict, Any
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

def delete_dir(dir_path: str) -> bool:
    """
    Utility to delete an empty or no empty folder

    Args:
        dir_path (str): The absolute path of the directory 

    Returns:
        Boolean: Returns True if the folder was found and deleted and false in case the folder was not found.
    """

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        return True
    else:
        return False

def get_cudaImgFromNumpy(img: np.ndarray) -> jetson_utils.cudaImage:
    """
    Convert a NumPy ndarray image to a Jetson-compatible CUDA image.

    Args:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        jetson_utils.cudaImage: The image converted to CUDA format for Jetson inference.
    """
    
    return jetson_utils.cudaFromNumpy(img)

def img_cudaResize(img: np.ndarray, width=300, height=300) -> jetson_utils.cudaImage:
    """
    Resize a NumPy ndarray image to a specific width and height using GPU-accelerated CUDA operations.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        width (int, optional): Target width of the resized image. Default is 300.
        height (int, optional): Target height of the resized image. Default is 300.

    Returns:
        jetson_utils.cudaImage: The resized image in CUDA format.
    """

    cuda_img_input = get_cudaImgFromNumpy(img)

    cuda_img_resized = jetson_utils.cudaAllocMapped(
        width=width,
        height=height,
        format=cuda_img_input.format
    )
    jetson_utils.cudaResize(cuda_img_input, cuda_img_resized)

    return cuda_img_resized

def get_str_from_dic(data: Dict[str, Any], key: str, default: str) -> str:
    """
    Retrieve a string value from a dictionary, converting the result if necessary.

    Args:
        data (Dict[str, Any]): A dictionary.
        key (str): The key to look for in the dictionary.
        default (str): The default string to return if the key is missing.

    Returns:
        str: The corresponding value as a string, or the default if the key is not found.
    """

    if key in data: return str(data[key])

    return default 

def get_int_from_dic(data: Dict[str, Any], key: str, default: int) -> int:
    """
    Retrieve an integer value from a dictionary, converting the result if necessary.

    Args:
        data (Dict[str, Any]): A dictionary.
        key (str): The key to look for in the dictionary.
        default (int): The default integer to return if the key is missing.

    Returns:
        int: The corresponding value converted to an integer, or the default if the key is not found.
    """

    if key in data: return int(data[key])

    return default

def get_float_from_dic(data: Dict[str, Any], key: str, default: float) -> float:
    """
    Retrieve a float value from a dictionary, converting the result if necessary.

    Args:
        data (Dict[str, Any]): A dictionary.
        key (str): The key to look for in the dictionary.
        default (int): The default float to return if the key is missing.

    Returns:
        float: The corresponding value converted to a float, or the default if conversion fails or the key is missing.
    """

    try:
        if key in data: return float(data[key])

        return default
    except (ValueError, TypeError):
        return default