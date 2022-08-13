import pymongo


client = pymongo.MongoClient(
    "****")

DB = client['face-recognition-db']


def models_to_create():
    """
    fetch updated classes, need to create a model for them.
    :return: list of classes names.
    """
    query = {"status": "updated"}
    connection = DB["classes_models"]
    results = [i["class_name"] for i in connection.find(query)]
    return results


def update_class_model_status(class_name, collection="classes_models"):
    connection = DB[collection]
    new_values = {"$set": {"status": "ready"}}
    query = {"class_name": class_name}
    return connection.update_one(query, new_values).modified_count

