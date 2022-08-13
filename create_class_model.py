import io
import face_recognition
from sklearn import svm
import pickle
import boto3
import tempfile


s3 = boto3.client(
    service_name='s3',
    region_name='eu-central-1',
    aws_access_key_id='******',
    aws_secret_access_key='*****'
)


def get_kids_images_names(class_name, kids_list):
    images_names = {}
    for kid in kids_list:
        list_objects = s3.list_objects(Bucket="classes-images", Prefix=f"{class_name}/{kid}")
        images_names[f"{kid}"] = []
        for key in list_objects["Contents"]:
            images_names[f"{kid}"].append(key["Key"])
    return images_names


def create_kids_names_list(class_name):
    list_objects = s3.list_objects(Bucket="classes-images", Delimiter='/', Prefix=f"{class_name}/")
    results = []
    for obj in list_objects['CommonPrefixes']:
        # path should be class_name/kid_id
        # so we need the second level.
        second_level_path = obj["Prefix"].split('/')[1]
        results.append(second_level_path)
    return results


def retrieve_all_data(class_name):
    """

    :param class_name:
    :return: dict containing all class kids as keys (including unknown key)
     and the value for each key is a list of their images keys (for S3)
    """
    kids_list = create_kids_names_list(class_name)
    kids_images_dict = get_kids_images_names(class_name, kids_list)
    # unknown_images_dict = get_unknown_images_names_list()
    # result_dict = {**kids_images_dict, **unknown_images_dict}
    # return result_dict
    return kids_images_dict


def get_unknown_images_names_list():
    unknown_names_dict = {"unknown": []}
    list_objects = s3.list_objects(Bucket="classes-images", Prefix=f"unknown/")
    for key in list_objects["Contents"]:
        unknown_names_dict["unknown"].append(key["Key"])
    return unknown_names_dict


def train_and_save_model(class_name, encodings, names):
    # Create and train the SVC classifier
    print(f'Creating {class_name} class model')
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)
    print('Done creating the model')

    # create an iterator object with write permission - model.pkl
    file_name = f"svm_pkl_{class_name}"
    print(f'Creating {class_name} class pickle file')
    with tempfile.TemporaryFile() as fp:
        pickle.dump(clf, fp)
        fp.seek(0)
        s3.upload_fileobj(io.BytesIO(fp.read()), Bucket='pickle-files-models', Key=file_name)


def build_svm_model(class_name):
    # The training data would be all the face encodings from all the known images
    # and the labels are their names
    encodings = []
    names = []
    s3_keys_per_kid = retrieve_all_data(class_name)
    print(f'Start encoding for {class_name} class')

    for kid, keys_list in s3_keys_per_kid.items():
        kid_encodings, labels = download_kid_images_and_encode(keys_list, kid)
        encodings = [*encodings, *kid_encodings]
        names = [*names, *labels]

    print(f'Done creating {class_name} class model')

    train_and_save_model(class_name, encodings, names)


def download_kid_images_and_encode(keys, kid):
    encodings = []
    names = []
    for key in keys:
        obj = s3.get_object(Bucket="classes-images", Key=key)
        file_stream = obj["Body"]
        encode = encode_img(file_stream, key)
        # Add face encoding for current image with corresponding label (name) to the training data
        encodings.append(encode)
        names.append(kid)
    return encodings, names


def encode_img(in_memory_file, key):
    face = face_recognition.load_image_file(in_memory_file)
    face_bounding_boxes = face_recognition.face_locations(face)
    # If training image contains exactly one face
    if len(face_bounding_boxes) == 1:
        face_enc = face_recognition.face_encodings(face)[0]
        return face_enc
    else:
        print(key + " was skipped and can't be used for training")


