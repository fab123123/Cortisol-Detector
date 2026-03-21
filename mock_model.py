import numpy as np


class MockCortisolModel:
    """
    Stand-in for your teammate's cortisol detection model.
    Mirrors the expected interface so you can swap it out with one line
    when the real file arrives.
    """

    def __init__(self, model_path=None):
        """
        model_path: path to a .h5 or .pkl file (ignored in mock).
        The real model will likely load weights here.
        """
        self.model_path = model_path
        print(f"[MockCortisolModel] Initialized (model_path={model_path})")

    def predict(self, image: np.ndarray) -> float:
        """
        Takes a 48x48 grayscale numpy array.
        Returns a float between 0.0 and 1.0 representing cortisol probability.

        Replace this class entirely with your teammate's when it's ready.
        """
        assert image.shape == (48, 48), f"Expected (48, 48), got {image.shape}"
        return 0.65  # Hardcoded dummy value