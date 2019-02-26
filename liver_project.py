from train import train
from predict import predict

'''Define Directories'''
main_path = 'C:/Users/user007/Desktop/shani/Liver_project/'

train_data_dir = main_path + 'TrainData/train/'
val_data_dir = main_path + 'TrainData/val/'
weights_path =  main_path + 'Weights/'

test_data_dir = main_path + 'TestData/'
out_seg_dir = main_path + 'testseg/'
weights_path_list = [weights_path + "/weights_lesion.pth", weights_path + "/weights_liver.pth"]

'''Define Training Mode or Testing Mode'''
# for train mode choose train_flag = True
# for test mode choose train_flag = False
train_flag = False

if train_flag:
    train(train_data_dir, val_data_dir, weights_path)
else:
    predict(test_data_dir, weights_path_list, out_seg_dir)

