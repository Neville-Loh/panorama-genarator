import pickle


def save_object_at_location(location, obj):
    with open(location, 'wb+') as file:
        # Step 3
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object_at_location(location):
    with open(location, 'rb+') as file:
        obj = pickle.load(file)

        # After config_dictionary is read from file
        return obj



