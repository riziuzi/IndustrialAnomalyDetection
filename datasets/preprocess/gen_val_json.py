import os
import glob
import json
import argparse


normal_mask_dir = "/home/medical/Anomaly_Project/InCTRL/data/"
def get_args_parser(objects_name):
    parser = argparse.ArgumentParser('AD dataset json generation', add_help=False)
    parser.add_argument('--dataset_dir', type=str, help='path to dataset dir',
                        default=directory)
    parser.add_argument('--output_dir', type=str, help='path to output dir',
                        default='./AD_json/')
    parser.add_argument('--obj_name', type=str, help='dataset name',
                        default='brainmri')
    parser.add_argument('--list_obj_name', type=list, default=objects_name)
    return parser.parse_args()

def list_objects(args):
    image_dir = os.path.join(args.dataset_dir, args.obj_name)
    save_path_normal = os.path.join(args.output_dir, args.dataset_dir.split('/')[-2].split("_")[0], args.obj_name+"_val_normal.json")
    save_path_outlier = os.path.join(args.output_dir, args.dataset_dir.split('/')[-2].split("_")[0], args.obj_name + "_val_outlier.json")
    print("Normal Path -> ",save_path_normal, "Outlier Path -> ",save_path_outlier)
    normal_data = list()
    normal_output_dict = list()

    normal_test_files = os.listdir(os.path.join(image_dir, 'test', 'good'))
    for file in normal_test_files:
        path_dict = {}
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
            normal_data.append('test' + '/good/' + file)
            path_dict["image_path"] = os.path.join(image_dir, 'test', 'good', file)
            path_dict["target"] = 0
            path_dict["type"] = args.obj_name
            path_dict["mask"] = os.path.join(normal_mask_dir,"normal_mask.PNG")
            normal_output_dict.append(path_dict)

    outlier_output_dict = list()
    outlier_data = list()
    outlier_data_dir = os.path.join(image_dir, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl in outlier_classes:
        if cl == 'good':
            continue
        outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
        for file in outlier_file:
            path_dict = {}
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'JPG' in file[-3:] or 'jpeg' in file[-4:]:
                outlier_data.append('test/' + cl + '/' + file)
                path_dict["image_path"] = os.path.join(image_dir, 'test', cl, file)
                path_dict["target"] = 1
                s = os.path.join(image_dir, 'ground_truth', cl, file)
                s = s[:-4] + "_mask" + s[-4:]
                if os.path.isfile(s):
                    path_dict["mask"] = s
                elif os.path.isfile(s[:-9]+".png"):
                    path_dict["mask"] = s[:-9]+".png"
                else: return 1
                path_dict["type"] = args.obj_name
                outlier_output_dict.append(path_dict)


    os.makedirs(os.path.dirname(save_path_normal), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_outlier), exist_ok=True)

    with open(save_path_normal, 'w') as normal_file:
        json.dump(normal_output_dict, normal_file)

    with open(save_path_outlier, 'w') as outlier_file:
        json.dump(outlier_output_dict, outlier_file)

if __name__ == "__main__":
    directories = [
            #   "/home/medical/Anomaly_Project/InCTRL/mvtec_anomaly_detection/",
            #   "/home/medical/Anomaly_Project/InCTRL/data/BrainMRI_anomaly_detection/",
                #    "/home/medical/Anomaly_Project/InCTRL/AITEX_anomaly_detection/",
                #    "/home/medical/Anomaly_Project/InCTRL/elpv_anomaly_detection/",
                #    "/home/medical/Anomaly_Project/InCTRL/SDD_anomaly_detection/",
                   "/home/medical/Anomaly_Project/InCTRL/data/visa_anomaly_detection/"                 # -> path updated, so need to create json again
    ]
    for directory in directories:
        objects_name = []
        for obj in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, obj)):
                objects_name.append(obj)
        args = get_args_parser(objects_name)
        if(args.list_obj_name):
            print("List of objects to be considered : ", args.list_obj_name)
            for obj in args.list_obj_name:
                # if(obj=='brainmri'):
                args.obj_name = obj
                temp = list_objects(args)
                if temp==1:
                    print("000000000000000")
        else:
            print("Object must be given in list !")


