import random, json
import pandas as pd
import numpy as np
from PIL import Image
from math import floor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, rotate
from torchvision.transforms import transforms

BATCH_SIZE = 32
path = '/home/aislab/Documents/Tellaroli/ProgettoSML/'
TRAINING_CSV    = path+'training_set.csv'
VALIDATION_CSV  = path+'validation_set.csv'
TEST_CSV        = path+'test_set.csv'
TRAIN_VALID_CSV = path+'training_validation_set.csv'
ANNOTATION_FILE = path+'annotation_file.csv'

def to_json(filename, dict):
    with open(filename, 'w') as fp:
        json.dump(dict, fp)

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

class slamDataset(Dataset):
    def __init__(self, csv, transform=None):
        super().__init__() 
        self.catalog = pd.read_csv(csv)
        self.transform = transform
    
    def __len__(self): return len(self.catalog)

    def __getitem__(self, index):
        row = self.catalog.iloc[index]
        img = to_tensor(Image.open(path+row['path']))
        if self.transform: img = self.transform(img)
        return img, row['area'], row['ate'], row['are']

class WrapperDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, area, ate, are = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, area, ate, are

    def __len__(self):
        return len(self.dataset)

resize = transforms.Resize(
    (224,224),
    antialias=True,
    interpolation=transforms.InterpolationMode.BICUBIC
)

class Random90Rotation:
    def __call__(self, x):
        angle = random.choice([0, 90, 180, 270])
        return rotate(x,angle)

train_transform = transforms.Compose([
    resize,
    Random90Rotation(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

def make_dataset(annotation_file=ANNOTATION_FILE, transform=resize):
    return slamDataset(annotation_file, transform=transform)

def make_train_val_set(annotation_file=TRAIN_VALID_CSV, transform=train_transform):
    return slamDataset(annotation_file, transform=transform)

def make_train_set(annotation_file=TRAINING_CSV, transform=train_transform):
    return slamDataset(annotation_file,transform=transform)

def make_test_set(annotation_file=TEST_CSV, transform=resize):
    return slamDataset(annotation_file,transform=transform)

def make_val_set(annotation_file=VALIDATION_CSV, transform=resize):
    return slamDataset(annotation_file,transform=transform)

def make_train_loader(
        annotation_file=TRAINING_CSV,
        transform=train_transform, 
        batch_size=BATCH_SIZE
    ):
    train_set = slamDataset(annotation_file,transform=transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)

def make_validation_loader(
        annotation_file=VALIDATION_CSV,
        transform=resize, 
        batch_size=BATCH_SIZE
    ):
    val_set = slamDataset(annotation_file,transform=transform)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False)

def make_test_loader(
        annotation_file=TEST_CSV,
        transform=resize, 
        batch_size=1
    ):
    test_set = make_test_set()
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)

def random_csv_split(
        perc = [.70,.15,.15],
        annotation_file = ANNOTATION_FILE,
        train_csv = TRAINING_CSV,
        val_csv = VALIDATION_CSV,
        test_csv = TEST_CSV,
        train_val_csv = TRAIN_VALID_CSV
    ):
    '''
    perc = [p1,p2,p3] : p1+p2+p3=1
    '''
    split_perc = np.array(perc)   #training, validation, test
    assert sum(split_perc)==1, "the perc sum must be 100"
    
    values = pd.read_csv(annotation_file)
    n = len(values)
    indexes = list(range(n))
    random.shuffle(indexes)
    a,b = floor(n*split_perc[0]), floor(n*split_perc[1])
    training = values.iloc[indexes[:a]]
    validation = values.iloc[indexes[a:a+b]]
    test = values.iloc[indexes[a+b:]]
    train_val = values.iloc[:a+b]
    
    training.to_csv(train_csv,index=False)
    validation.to_csv(val_csv,index=False)
    test.to_csv(test_csv,index=False)
    train_val.csv(train_val_csv,index=False)

def load_data():
    return make_train_set(), make_test_set()