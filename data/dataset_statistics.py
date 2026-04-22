
# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]
MEANS['tinyimagenet'] = [0.485, 0.456, 0.406]
STDS['tinyimagenet'] = [0.229, 0.224, 0.225]


# ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']
MEANS['imagenette'] = [0.485, 0.456, 0.406]
STDS['imagenette'] = [0.229, 0.224, 0.225]
MEANS['imagewoof'] = [0.485, 0.456, 0.406]
STDS['imagewoof'] = [0.229, 0.224, 0.225]
MEANS['imagemeow'] = [0.485, 0.456, 0.406]
STDS['imagemeow'] = [0.229, 0.224, 0.225]
MEANS['imagesquawk'] = [0.485, 0.456, 0.406]
STDS['imagesquawk'] = [0.229, 0.224, 0.225]
MEANS['imagefruit'] = [0.485, 0.456, 0.406]
STDS['imagefruit'] = [0.229, 0.224, 0.225]
MEANS['imageyellow'] = [0.485, 0.456, 0.406]
STDS['imageyellow'] = [0.229, 0.224, 0.225]

# MedMNIST datasets.
# Common local runs use exact train+val statistics computed after ToTensor() -> [0, 1].
# Remaining entries are kept as existing approximations until explicitly recomputed.
MEANS['pathmnist'] = [0.740638, 0.533137, 0.705940]
STDS['pathmnist'] = [0.123628, 0.176805, 0.124425]

MEANS['chestmnist'] = [0.5096]
STDS['chestmnist'] = [0.2751]

MEANS['dermamnist'] = [0.763445, 0.538597, 0.562011]
STDS['dermamnist'] = [0.136320, 0.153896, 0.168832]

MEANS['octmnist'] = [0.2462]
STDS['octmnist'] = [0.1966]

MEANS['pneumoniamnist'] = [0.4784]
STDS['pneumoniamnist'] = [0.2157]

MEANS['retinamnist'] = [0.5333, 0.4667, 0.4588]
STDS['retinamnist'] = [0.1608, 0.1725, 0.1804]

MEANS['breastmnist'] = [0.5176]
STDS['breastmnist'] = [0.2510]

MEANS['bloodmnist'] = [0.794354, 0.659687, 0.696210]
STDS['bloodmnist'] = [0.215488, 0.241463, 0.117844]

MEANS['tissuemnist'] = [0.5804]
STDS['tissuemnist'] = [0.2431]

MEANS['organamnist'] = [0.5686]
STDS['organamnist'] = [0.2627]

MEANS['organcmnist'] = [0.5569]
STDS['organcmnist'] = [0.2706]

MEANS['organsmnist'] = [0.5608]
STDS['organsmnist'] = [0.2667]
