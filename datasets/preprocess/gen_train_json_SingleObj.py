import os
import glob
import pickle
import json
import argparse


def main(normal_path, anomaly_path, obj_name="Unknown"):
    save_path = os.path.join(output_dir, obj_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path_normal = os.path.join(output_dir,save_path, obj_name + "_normal.json")
    save_path_outlier = os.path.join(output_dir, save_path, obj_name + "_outlier.json")
    print(save_path_normal, save_path_outlier)

    normal_output_dict = list()
    for file in os.listdir(normal_path):
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'JPG' in file[-3:] or 'npy' in file[-3:]:
            path_dict["image_path"] = os.path.join(normal_path, file)
            path_dict["target"] = 0
            path_dict["type"] = obj_name
            normal_output_dict.append(path_dict)

    outlier_output_dict = list()

    for file in os.listdir(anomaly_path):
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
            path_dict["image_path"] = os.path.join(anomaly_path, file)
            path_dict["target"] = 1
            path_dict["type"] = obj_name
            outlier_output_dict.append(path_dict)

    os.makedirs(os.path.dirname(save_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_outlier), exist_ok=True)

    with open(save_path_normal, 'w') as normal_file:
        json.dump(normal_output_dict, normal_file)

    with open(save_path_outlier, 'w') as outlier_file:
        json.dump(outlier_output_dict, outlier_file)
    
    return (save_path_normal, save_path_outlier)
    

def save_list_to_file(my_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(my_list, file)

if __name__ == "__main__":
    normal_path = "/home/medical/Anomaly_Project/InCTRL/visa_anomaly_detection/pcb2/train/good/"
    # normal_path = "/home/medical/Anomaly_Project/InCTRL/mvtec_anomaly_detection/bottle/train/good/"
    # anomaly_path = "/home/medical/Anomaly_Project/pytorch-cutpaste/AD_json_train_CutPaste/"
    anomaly_path = "/home/medical/Anomaly_Project/pytorch-cutpaste/AD_json_train_CutPaste/visa_pcb2/"
    # anomaly_path = "/home/medical/Anomaly_Project/InCTRL/visa_anomaly_detection/pcb2/test/defect/"
    obj_name = "Visa_pcb2"

    # OUTPUTS
    output_dir = "/home/medical/Anomaly_Project/pytorch-cutpaste/SingleObjJSON/"
    file_path = "/home/medical/Anomaly_Project/pytorch-cutpaste/single_object_list_pcb2.pkl"





    lst = []
    lst.append(main(normal_path, anomaly_path, obj_name))
    
    # file_path = "/home/medical/Anomaly_Project/InCTRL/train_object_list.pkl"
    save_list_to_file(lst, file_path)
