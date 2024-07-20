import os
import glob
import pickle
import json
import argparse
import numpy as np

def get_args_parser(dataset_name):
    parser = argparse.ArgumentParser('AD dataset json generation', add_help=False)
    parser.add_argument('--dataset_dir', type=str, help='path to dataset dir',
                        default='/home/medical/Anomaly_Project/InCTRL/data')
    parser.add_argument('--output_dir', type=str, help='path to output dir',
                        default='/home/medical/Anomaly_Project/InCTRL/AD_json_train_new/')
    parser.add_argument('--dataset_name', type=str, help='dataset name',
                        default=dataset_name)
    return parser.parse_args()


def main(dataset_name, obj_name, unified_datasets=False, unified_objects=False):
    args = get_args_parser(dataset_name)
    image_dir = os.path.join(args.dataset_dir, args.dataset_name)
    save_path = None
    if(unified_datasets): save_path = args.output_dir
    elif(unified_objects): save_path = os.path.join(args.output_dir, args.dataset_name)
    else: save_path = save_path = os.path.join(args.output_dir, args.dataset_name+ "_" + obj_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path_normal = os.path.join(args.output_dir,save_path, args.dataset_name+ "_" + obj_name + "_normal.json")
    save_path_outlier = os.path.join(args.output_dir, save_path, args.dataset_name + "_" + obj_name + "_outlier.json")
    # print(save_path_normal, save_path_outlier)

    normal_data = list()
    normal_output_dict = list()
    normal_train_files = os.listdir(os.path.join(image_dir,obj_name, 'train', 'good'))
    for file in normal_train_files:
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            normal_data.append('train' + '/good/' + file)
            path_dict["image_path"] = os.path.join(image_dir,obj_name, 'train', 'good', file)
            path_dict["target"] = 0
            path_dict["mask"] = os.path.join(args.dataset_dir,"normal_mask.PNG")
            path_dict["type"] = obj_name
            normal_output_dict.append(path_dict)

    normal_test_files = os.listdir(os.path.join(image_dir,obj_name, 'test', 'good'))                # why this into the training file?
    for file in normal_test_files:
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
            normal_data.append('test' + '/good/' + file)
            path_dict["image_path"] = os.path.join(image_dir,obj_name, 'test', 'good', file)
            path_dict["target"] = 0
            path_dict["mask"] = os.path.join(args.dataset_dir,"normal_mask.PNG")
            path_dict["type"] = obj_name
            normal_output_dict.append(path_dict)

    outlier_output_dict = list()
    outlier_data = list()
    outlier_data_dir = os.path.join(image_dir,obj_name, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl in outlier_classes:
        if cl == 'good':
            continue
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
        for file in outlier_file:
            path_dict = {}
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
                outlier_data.append('test/' + cl + '/' + file)
                path_dict["image_path"] = os.path.join(image_dir,obj_name, 'test', cl, file)
                path_dict["target"] = 1
                s = os.path.join(image_dir,obj_name, 'ground_truth', cl, file)
                s = s[:-4] + "_mask" + s[-4:]
                if os.path.isfile(s):
                    path_dict["mask"] = s
                elif os.path.isfile(s[:-9]+".png"):
                    path_dict["mask"] = s[:-9]+".png"
                else: return
                path_dict["type"] = obj_name
                outlier_output_dict.append(path_dict)

    os.makedirs(os.path.dirname(save_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_outlier), exist_ok=True)

    with open(save_path_normal, 'w') as normal_file:
        json.dump(normal_output_dict, normal_file)

    with open(save_path_outlier, 'w') as outlier_file:
        json.dump(outlier_output_dict, outlier_file)
    
    return [[save_path_normal], [save_path_outlier]]
    

def save_list_to_file(my_list, file_path):
    with open(file_path, 'wb') as file:  # Use 'wb' for writing in binary mode
        pickle.dump(my_list, file)

def generate(unified_datasets=False,unified_objects=False,directories=None):
    if unified_datasets:
        lst = [[[],[]]]
    elif unified_objects:
        lst = []
    else:
        lst=[]
    
    if unified_objects:
        cnt = 0
        path_index = {}
        for i in directories:
            path_index[i.split("/")[-2]] = cnt
            cnt+=1
        for i in range(cnt):
            lst.append([[],[]])
    objects_path = []
    for directory in directories:
        for folder_name in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, folder_name)):
                objects_path.append(os.path.join(directory, folder_name))
        # print("List of objects to be considered : ", objects_path)
    for obj_path in objects_path:
        results = main(obj_path.split('/')[-2], obj_path.split('/')[-1],unified_datasets=True)
        if results:
            if unified_datasets:
                lst[0][0].append(results[0][0])
                lst[0][1].append(results[1][0])
            elif unified_objects:
                lst[path_index[obj_path.split("/")[-2]]][0].append(results[0][0])
                lst[path_index[obj_path.split("/")[-2]]][1].append(results[1][0])
            else:
                lst.append(results)
        else:
            print(obj_path)
            print("OOOOOOOOOOOOOOOOOOOOOOOO")
    
    file_path = "/home/medical/Anomaly_Project/InCTRL/data/train_object_list.pkl"
    save_list_to_file(lst, file_path)


if __name__ == '__main__':
    directories = [
                 "/home/medical/Anomaly_Project/InCTRL/data/visa_anomaly_detection/",                  # -> path updated, so need to create json again
                   "/home/medical/Anomaly_Project/InCTRL/data/mvtec_anomaly_detection/",
                 "/home/medical/Anomaly_Project/InCTRL/data/AITEX_anomaly_detection/",               # NO: will be used for validation (disjoint from training dataset); YES: Medical dataset will be used for the validation
                 "/home/medical/Anomaly_Project/InCTRL/data/elpv_anomaly_detection/",                   # it does not have ground truth
                 "/home/medical/Anomaly_Project/InCTRL/data/SDD_anomaly_detection/",
    ]
    generate(unified_datasets=True,directories=directories)