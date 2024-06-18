import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA


class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        """
        origin data
        """
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth

    @property
    def normal_cube(self):
        """
        normalization data: range(0, 1)
        """
        return (self.data_cube - np.min(self.data_cube)) / (np.max(self.data_cube) - np.min(self.data_cube))
        #return self.data_cube


class PaviaURaw(DataReader):
    def __init__(self):
        super(PaviaURaw, self).__init__()

        raw_data_package = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/paviaU1.mat")
        self.data_cube = raw_data_package["paviaU1"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/paviaU1_gt.mat")
        self.g_truth = truth["paviaU1_gt"].astype(np.float32)

class XARaw(DataReader):
    def __init__(self):
        super(XARaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Xiongan.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Xiongan_GT.mat")
        self.g_truth = truth["gt"].astype(np.float32)
class DioniRaw(DataReader):
    def __init__(self):
        super(DioniRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Dioni.mat")
        self.data_cube = raw_data_package["Dioni"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Dioni_GT.mat")
        self.g_truth = truth["Dioni_GT"].astype(np.float32)
class Houston2018trRaw(DataReader):
    def __init__(self):
        super(Houston2018trRaw, self).__init__()

        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston_2013.mat")
        self.data_cube = raw_data_package["Houston_2013"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/TRLabel.mat")
        self.g_truth = truth["TRLabel"].astype(np.float32)
class testtrRaw(DataReader):
    def __init__(self):
        super(testtrRaw, self).__init__()

        raw_data_package = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/test.mat")
        self.data_cube = raw_data_package["test"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/test_gt.mat")
        self.g_truth = truth["test_gt"].astype(np.float32)
class testRaw(DataReader):
    def __init__(self):
        super(testRaw, self).__init__()

        raw_data_package = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/test1.mat")
        self.data_cube = raw_data_package["test1"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/test_gt1.mat")
        self.g_truth = truth["test_gt1"].astype(np.float32)
class Houston2018teRaw(DataReader):
    def __init__(self):
        super(Houston2018teRaw, self).__init__()

        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston_2013.mat")
        self.data_cube = raw_data_package["Houston_2013"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/TSLabel.mat")
        self.g_truth = truth["TSLabel"].astype(np.float32)
class HanchuangRaw(DataReader):
    def __init__(self):
        super(HanchuangRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/HanChuan.mat")
        self.data_cube = raw_data_package["HanChuan"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/HanChuan_gt.mat")
        self.g_truth = truth["HanChuan_gt"].astype(np.float32)
class LouKiaRaw(DataReader):
    def __init__(self):
        super(LouKiaRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Loukia.mat")
        self.data_cube = raw_data_package["Loukia"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/HyperLargeModel/HyperLargeModel/data/Loukia_GT.mat")
        self.g_truth = truth["Loukia_GT"].astype(np.float32)
class IndianRaw(DataReader):
    def __init__(self):
        super(IndianRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/Indian_pines_corrected.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/Indian_pines_gt.mat")
        self.g_truth = truth["groundT"].astype(np.float32)
class indian_merge_labelRaw(DataReader):
    def __init__(self):
        super(indian_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/Indian_pines_gt_merge.mat")
        self.g_truth = truth["groundTM"].astype(np.float32)

class HoustonRaw(DataReader):
    def __init__(self):
        super(HoustonRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston.mat")
        self.data_cube = raw_data_package["Houston"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston_GT.mat")
        self.g_truth = truth["Houston_GT"].astype(np.float32)
class Houston_merge_labelRaw(DataReader):
    def __init__(self):
        super(Houston_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston_GT_merge.mat")
        self.g_truth = truth["Houston_GT_merge"].astype(np.float32)

class HongHuRaw(DataReader):
    def __init__(self):
        super(HongHuRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/HongHu.mat")
        self.data_cube = raw_data_package["HongHu"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/HongHu_gt.mat")
        self.g_truth = truth["HongHu_gt"].astype(np.float32)
class Houston_merge_labelRaw(DataReader):
    def __init__(self):
        super(Houston_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/Houston_GT_merge.mat")
        self.g_truth = truth["Houston_GT_merge"].astype(np.float32)

class HongHu_subRaw(DataReader):
    def __init__(self):
        super(HongHu_subRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/HongHu_sub.mat")
        self.data_cube = raw_data_package["HongHu_sub"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/HongHu_sub_gt.mat")
        self.g_truth = truth["HongHu_sub_gt"].astype(np.float32)
class HongHu_sub_merge_labelRaw(DataReader):
    def __init__(self):
        super(HongHu_sub_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/HongHu_sub_gt_merge.mat")
        self.g_truth = truth["HongHu_sub_gt_merge"].astype(np.float32)

class LongKouRaw(DataReader):
    def __init__(self):
        super(LongKouRaw, self).__init__()
        raw_data_package = sio.loadmat(r"/data/yao.jin/CNN/dataset/LongKou.mat")
        self.data_cube = raw_data_package["LongKou"].astype(np.float32)
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/LongKou_gt.mat")
        self.g_truth = truth["LongKou_gt"].astype(np.float32)
class LongKouRaw_merge_labelRaw(DataReader):
    def __init__(self):
        super(LongKouRaw_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/CNN/dataset/LongKou_gt_merge.mat")
        self.g_truth = truth["LongKou_gt_merge"].astype(np.float32)


class pavia_merge_labelRaw(DataReader):
    def __init__(self):
        super(pavia_merge_labelRaw, self).__init__()
        truth = sio.loadmat(r"/data/yao.jin/WFCG/WFCG-master/Datasets/paviaU1_gt_merge.mat")
        self.g_truth = truth["paviaU1_gt_merge"].astype(np.float32)

class trainindex(DataReader):
    def __init__(self):
        super(trainindex, self).__init__()
        truth = sio.loadmat(r"G:\CUG\code\python\WFCG\WFCG-master\datasets\trainindex.mat")
        self.g_truth = truth["Trainindex"].astype(np.float32)



class KSCRaw(DataReader):

    def __init__(self):
        super(KSCRaw, self).__init__()

        raw_data_package = sio.loadmat("/data/yao.jin/CNN/dataset/KSC.mat")
        self.data_cube = raw_data_package["KSC"].astype(np.float32)
        truth = sio.loadmat("/data/yao.jin/CNN/dataset/KSC_gt.mat")
        self.g_truth = truth["KSC_gt"]
class KSC_teRaw(DataReader):

    def __init__(self):
        super(KSC_teRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\KSC_te.mat")
        self.data_cube = raw_data_package["KSC_te"].astype(np.float32)
        truth = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\KSC_te_gt.mat")
        self.g_truth = truth["KSC_te_gt"]
class KSC_trRaw(DataReader):

    def __init__(self):
        super(KSC_trRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\KSC_tr.mat")
        self.data_cube = raw_data_package["KSC_tr"].astype(np.float32)
        truth = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\KSC_tr_gt.mat")
        self.g_truth = truth["KSC_tr_gt"]
class dataRaw(DataReader):

    def __init__(self):
        super(dataRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\data.mat")
        self.data_cube = raw_data_package["data"].astype(np.float32)
        truth = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\data_gt.mat")
        self.g_truth = truth["data_gt"]
class XTeRaw(DataReader):

    def __init__(self):
        super(XTeRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\PU\XTe.mat")
        self.data_cube = raw_data_package["XTe"].astype(np.float32)
        truth = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\PU\XTe_gt.mat")
        self.g_truth = truth["XTe_gt"]

class XTrRaw(DataReader):

    def __init__(self):
        super(XTrRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\PU\XTr.mat")
        self.data_cube = raw_data_package["XTr"].astype(np.float32)
        truth = sio.loadmat(r"E:\地大\代码\matlab\CNN\dataset\indian\XTr_gt.mat")
        self.g_truth = truth["XTr_gt"]


class testindexRaw(DataReader):

    def __init__(self):
        super(testindexRaw, self).__init__()

        truth = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\testindex.mat")
        self.g_truth = truth["testindex"]
       
class trainindexRaw(DataReader):

    def __init__(self):
        super(trainindexRaw, self).__init__()

        truth = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\trainindex.mat")
        self.g_truth = truth["trainindex"]

class data_diffRaw(DataReader):

    def __init__(self):
        super(data_diffRaw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\data_diff.mat")
        self.data_cube = raw_data_package["data_diff"].astype(np.float32)

class data_diff_2Raw(DataReader):

    def __init__(self):
        super(data_diff_2Raw, self).__init__()

        raw_data_package = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\data_diff_2.mat")
        self.data_cube = raw_data_package["data_diff_2"].astype(np.float32)
class waveRaw(DataReader):

    def __init__(self):
        super(waveRaw, self).__init__()

        truth = sio.loadmat(r"E:\CUG\code\matlab\CNN\dataset\indian\indian_wavelength.mat")
        self.g_truth = truth["indian_wavelength"]



# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca


def data_info(train_label=None, val_label=None, test_label=None, start=1):
    class_num = np.max(train_label.astype('int32'))
    if train_label is not None and val_label is not None and test_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i], "\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)

    elif train_label is not None and val_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)

    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)

    else:
        raise ValueError("labels are None")


def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 400, save_img=None):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)


if __name__ == "__main__":
    data = IndianRaw().cube
    data_gt = IndianRaw().truth
    IndianRaw().data_info(data_gt)
    IndianRaw().draw(data_gt, save_img=None)
    print(data.shape)
    print(data_gt.shape)
