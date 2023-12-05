def check_keras_3():
    """Raises an error if the version of Keras is less than 3."""
    import keras
    keras_version = keras.__version__

    if int(keras_version.split(".")[0]) < 3:
        raise ValueError(f"Keras version must be greater than 3. Found: {keras_version}. please run `pip install keras --upgrade` and restart.")

    print(f"Keras version: {keras_version}")