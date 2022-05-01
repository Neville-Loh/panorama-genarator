import pickle
import os


def save_object_at_location(location, obj):
    with safe_open_w(location) as file:
        # Step 3
        print(f'[INFO] Writing result as cache to %s' % location)
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object_at_location(location):
    with open(location, 'rb+') as file:
        obj = pickle.load(file)
        print(f'[INFO] Loading result as cache from %s' % location)
        # After config_dictionary is read from file
        return obj


def safe_open_w(path):
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')


def get_file_name_from_path(path):
    return os.path.split(path)[-1].split(".")[0]
