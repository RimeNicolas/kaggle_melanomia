class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 2
    metric = 'linear'
    easy_margin = False
    loss = 'bce'

    dataset_type = 'images'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train.csv'
    train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset3\jpeg_train\train.csv'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\train.csv'
    split_train_val = True

    optimizer = 'adam'
    use_gpu = True
    gpu_id = '0'
    num_workers = 4  

    input_shape = (3, 512, 512)
    train_batch_size = 32
    test_batch_size = train_batch_size 

    epoch_max = 10
    lr = 1e-4  # initial learning rate
    lr_step = 10 
    weight_decay = 5e-2

    checkpoints_path = r'C:\Users\Nrime\source\repos\kaggle_melanomia\arcface-pytorch\checkpoints'
    epoch_start = 0
    path_model_parameters = backbone + '_' + str(epoch_start) + '.pth'
    path_metric_parameters = 'metric' + '_' + str(epoch_start) + '.pth'
    save_interval = 5

    path_model_parameters_test = 'resnet18_10.pth'
    path_metric_parameters_test = 'metric_10.pth'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\submission.csv'
    test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\test.csv'
    test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\submission.csv'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test\test.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset2\jpeg_test\submission.csv'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\test_unlabelled.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\bench2\submission.csv'


