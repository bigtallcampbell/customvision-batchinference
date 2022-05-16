"""
Strong type the app config
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    """
    URL for the project
    """
    CUSTOM_VISION_ENDPOINT: str = ""

    """
    ID of the Custom Vision Project
    """
    CUSTOM_VISION_PROJECT_ID: str = ""


    """
    Training Key
    """
    CUSTOM_VISION_KEY: str = ""


    """
    Name of the published training iteration to run inference on
    """
    CUSTOM_VISION_PUBLISHED_ITERATION_NAME: str = ""


    """
    Source folder of images to process
    """
    SOURCE_FOLDER: str = ""


    """
    Minimum threshold to be considered a hit
    """
    PROBABILIY_THRESHOLD: float = 0.5


    #Remove python's scaffolding of the init so we don't have to force default values
    def __init__(self):
        pass