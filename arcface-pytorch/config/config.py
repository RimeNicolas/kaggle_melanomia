# For 224x224

class Config(object):
    env = 'default'
    backbone = 'efficientnet_b0'
    classify = 'softmax'
    num_classes = 2
    metric = 'linear'
    arc_s = 3
    arc_m = 0.0
    easy_margin = False
    loss = 'focal_loss' # seems to give worse results for training accuracy but better for validation

    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train.csv'
    train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\train.csv'

    optimizer = 'adam'
    use_gpu = True
    gpu_id = '0'
    num_workers = 4  

    input_shape = (3, 224, 224)
    train_batch_size = 128
    test_batch_size = train_batch_size 

    n_fold = 3 # 0 is all dataset
    epoch_max = 30
    lr_model = 1e-4 
    lr_metric = 2e-2 
    # lr_step = 10 # divide lr by 10 every lr_step (removed atm)
    # weight_decay = 2e-1 # for resize img only
    weight_decay = 2e-1
    accuracy_val_interval = 1
    accuracy_train_interval = 10 * accuracy_val_interval

    save_model = True
    checkpoints_path = r'C:\Users\Nrime\source\repos\kaggle_melanomia\arcface-pytorch\checkpoints'
    epoch_start = 0
    # path_model_parameters = backbone + '_' + str(epoch_start) + '.pth'
    # path_metric_parameters = 'metric' + '_' + str(epoch_start) + '.pth'
    save_interval = 1

    # path_model_parameters_test = backbone + '_' + str(epoch_max) + '.pth'
    # path_metric_parameters_test = 'metric' + '_' + str(epoch_max) + '.pth'
    path_model_parameters_test = backbone + '.pth'
    path_metric_parameters_test = 'metric' + '.pth'
    test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\testset\test.csv'
    test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\testset\submission.csv'



    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\submission.csv'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test\test.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test\submission.csv'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\test_unlabelled.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\submission.csv'


