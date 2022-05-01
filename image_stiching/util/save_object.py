import pickle
import os



def save_object_at_location(location, obj):
    with open(location, 'wb+') as file:
        # Step 3
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object_at_location(location):
    with open(location, 'rb+') as file:
        obj = pickle.load(file)

        # After config_dictionary is read from file
        return obj


def get_file_name_from_path(path):
    return os.path.split(path)[-1].split(".")[0]


