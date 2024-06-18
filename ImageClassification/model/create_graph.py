import numpy as np

def get_label(gt_reshape, train_index, val_index, test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_index)):
        val_samples_gt[val_index[i]] = gt_reshape[val_index[i]]

    return train_samples_gt, test_samples_gt, val_samples_gt

def label_to_one_hot(data_gt, class_num):

    height, width = data_gt.shape
    ont_hot_label = [] 
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_num, dtype=np.int64)
            if data_gt[i, j] != 0:
                temp[int(data_gt[i, j]) - 1] = 1
            ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [height * width, class_num])
    return ont_hot_label


def get_label_mask(train_samples_gt, test_samples_gt, val_samples_gt, data_gt, class_num):
    
    height, width = data_gt.shape
    # train
    train_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num]) 
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_num])

    # test
    test_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    # test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_num])

    # val
    val_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    # val_samples_gt = np.reshape(val_samples_gt, [height * width])
    for i in range(height * width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height * width, class_num])

    return train_label_mask, test_label_mask, val_label_mask