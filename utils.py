import os
import importlib


def import_model(model_path):
    """
    import model from file path
    """
    model_file = os.path.basename(model_path).split(".")[0]
    model = importlib.import_module("models.%s" % model_file)
    return model