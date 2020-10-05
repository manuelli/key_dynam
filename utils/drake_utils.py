# system
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

# pydrake
from pydrake.multibody.tree import BodyIndex


def get_label_db(plant,  # pydrake MultiBodyPlant
                 ):
    """
    Builds database that associates bodies and labels
    :return: TinyDB database
    :rtype:
    """

    db = TinyDB(storage=MemoryStorage)
    for i in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(i))
        model_instance_id = int(body.model_instance())

        body_name = body.name()
        model_name = plant.GetModelInstanceName(body.model_instance())

        entry = {'label': i,
                 'model_instance_id': model_instance_id,
                 'body_name': body_name,
                 'model_name': model_name}
        db.insert(entry)

    return db