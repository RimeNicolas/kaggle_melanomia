class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 2
    metric = 'None'
    easy_margin = False
    loss = 'focal_loss'

    dataset_type = 'images'
    # train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_train\train.csv'
    train_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\train.csv'
    split_train_val = False

    optimizer = 'adam'
    use_gpu = True
    gpu_id = '0'
    num_workers = 4  

    input_shape = (3, 256, 256)
    train_batch_size = 64
    test_batch_size = 4

    epoch_max = 20
    lr = 1e-4  # initial learning rate
    lr_step = 10
    weight_decay = 5e-4

    checkpoints_path = r'C:\Users\Nrime\source\repos\kaggle_melanomia\arcface-pytorch\checkpoints'
    epoch_start = 0
    path_model_parameters = backbone + '_' + str(epoch_start) + '.pth'
    save_interval = 10

    path_model_parameters_test = 'resnet18_3.pth'
    # test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\test.csv'
    # test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\dataset1\jpeg_test\submission.csv'

    test_list = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\test.csv'
    test_save = r'C:\Users\Nrime\Documents\Kaggle_dataset\melanoma\original\jpeg\submission.csv'

