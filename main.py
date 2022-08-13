from mongo_quries import models_to_create, update_class_model_status
from create_class_model import build_svm_model

"""
Fetch from classes_models all models to be calculated (status == updated).
for each class ->
    fetch from S3 unknown images and class students images
    encode all the images
    load the encoded images with labels to SVM model 
    save the model to in-memory pickle file and send it to S3
    update the classes_models current document (status == ready)
"""
print('starting the caclulations')
classes_to_calculate = models_to_create()
print("classes to calculate:")
for class_name in classes_to_calculate:
    print(f"{class_name}")
    build_svm_model(class_name)
    print(f"for {class_name}: ", update_class_model_status(class_name))
