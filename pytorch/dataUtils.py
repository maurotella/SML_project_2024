import random, json
import pandas as pd
import numpy as np
import random
from PIL import Image
from math import floor
from torch import Tensor
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

ROTTE = [
    'E53-0', 'E23-3', '26-0', '36-8', '0510035520_A_40_1_103','0510045922_A-40.1-030',
    'NE49-4', 'W7-3', '36-0', 'W20-0', '12-1','4-0', '2-0', '2-2', '32-G7',
    '0510045904_A_40_1_102', '56-7','16-5', '37-1'
]

def to_json(filename, dict):
    with open(filename, 'w') as fp:
        json.dump(dict, fp)

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

def load_map(path) -> Tensor:
    return to_tensor(Image.open(path))

extract_name = lambda path: path.split('/')[1]

class slamDataset(Dataset):
    def __init__(self, csv, transform=None, broken=True, only_broken=False):
        super().__init__() 
        self.catalog = pd.read_csv(csv)
        if not broken:
            self.catalog = self.catalog[self.catalog['path'].apply(extract_name).apply(lambda x: x not in ROTTE)].copy()
        if only_broken:
            assert broken
            self.catalog = self.catalog[self.catalog['path'].apply(extract_name).apply(lambda x: x in ROTTE)].copy()
        self.transform = transform
    
    def __len__(self): return len(self.catalog)

    def __getitem__(self, index):
        row = self.catalog.iloc[index]
        img = load_map(path+row['path'])
        if self.transform: img = self.transform(img)
        return img, row['area'], row['ate'], row['are']

class IncreaseSlamDataSet(Dataset):
    def __init__(self, slamdataset, perc, broken=True):
        super().__init__()
        assert perc>=0
        self.l = len(slamdataset)
        self.len = self.l+round(self.l*perc)
        self.dataset = slamdataset
        self.extra = dict()
    
    def __len__(self): return self.len

    def __getitem__(self, index):
        if index<self.l: return self.dataset[index]
        if index not in self.extra:
            self.extra[index] = random.randint(0,self.l-1)
        return self.dataset[self.extra[index]]

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
    #transforms.Normalize(0.5,0.5)
])

def make_dataset(annotation_file=ANNOTATION_FILE, transform=resize, broken=True, only_broken=False):
    return slamDataset(annotation_file, transform=transform, broken=broken, only_broken=only_broken)

def make_train_val_set(annotation_file=TRAIN_VALID_CSV, transform=resize):
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
        batch_size=BATCH_SIZE,
        increase_perc=0,
        broken=True,
        only_broken=False
    ):
    train_set = slamDataset(annotation_file,transform=transform, broken=broken, only_broken=only_broken)
    if increase_perc>0: 
        train_set = IncreaseSlamDataSet(train_set,increase_perc, broken=broken, only_broken=only_broken)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

def make_validation_loader(
        annotation_file=VALIDATION_CSV,
        transform=resize, 
        batch_size=BATCH_SIZE,
        increase_perc=0,
        broken=True
    ):
    val_set = slamDataset(annotation_file,transform=transform, broken=broken)
    if increase_perc>0: 
        val_set = IncreaseSlamDataSet(val_set,increase_perc, broken=broken)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

def make_test_loader(
        annotation_file=TEST_CSV,
        transform=resize, 
        batch_size=1,
        increase_perc=0,
        broken=True,
        only_broken=False
    ):
    test_set = slamDataset(annotation_file,transform=transform, broken=broken, only_broken=only_broken)
    if increase_perc>0: 
        test_set = IncreaseSlamDataSet(test_set,increase_perc, broken=broken, only_broken=only_broken)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

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
    train_val.to_csv(train_val_csv,index=False)

def load_data():
    return make_train_set(), make_test_set()

data_info = load_json('data_info.json')
def revert_data_norm(ate,are):
    return np.expm1(ate*data_info['max_log_ate']),np.expm1(are*data_info['max_log_are'])