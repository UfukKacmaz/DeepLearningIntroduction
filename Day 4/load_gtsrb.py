import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import six.moves.cPickle as pickle
import numpy as np
import cv2
import os.path
 
np.random.seed(42)

def load_gtsrb_images(dataset_path, classes = range(43), max_num_imgs_per_class=float('Inf') ):
    
    img_size = [32, 32]
    image_block = np.zeros( img_size + [3, 0] )
    labels = np.zeros( 0, dtype=np.int32 )
    sign_ids = np.zeros( 0 )
    sign_id = 0
        
    for class_idx in classes:
        image_block_class = np.zeros( img_size + [3, 0] )
        
        dump_file = dataset_path + ("gtsrb_%02d.dump" % class_idx)
        if not os.path.exists(dump_file):
            seq_idx = 0
            sub_path = dataset_path + ("/%05d/"  % class_idx)
            next_image = None
            img_idx = 1
            img_cnt = 0
            while (img_idx != 0):
                img_idx = 0
                sign_id = sign_id + 1
                while (True):
                    image_file = sub_path + ("%05d_%05d.ppm"  % (seq_idx, img_idx) )
                    print("Reading file: " + image_file) 
                    next_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                    img_cnt += 1
                    print(img_cnt)
                    if not next_image is None:
                        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
                        next_image = cv2.resize(next_image, tuple(img_size), interpolation = cv2.INTER_CUBIC)
                        image_block_class = np.concatenate( [image_block_class, np.expand_dims(next_image, 3)], axis=3)
                        sign_ids = np.concatenate( [sign_ids, [sign_id] ], axis=0 )
                    else:
                        break
                    img_idx += 1
                seq_idx += 1
                
            print("Serializing images", image_block_class.shape)
            file = open(dump_file, 'wb')
            pickle.dump(image_block_class, file)
            file.close()

    class_descs = []
    class_descs.append('Speed limit 20') #0
    class_descs.append('Speed limit 30') #1
    class_descs.append('Speed limit 50') #2
    class_descs.append('Speed limit 60') #3
    class_descs.append('Speed limit 70') #4
    class_descs.append('Speed limit 80') #5
    class_descs.append('Derestriction 80') #6 
    class_descs.append('Speed limit 100')
    class_descs.append('Speed limit 120') #8
    class_descs.append('Prohibit Overtaking')
    class_descs.append('Prohibit Overtaking for trucks')
    class_descs.append('Right of way on next intersection')
    class_descs.append('Right of way on this street') #12 
    class_descs.append('Yield way') #13
    class_descs.append('Stop') #14
    class_descs.append('No entry') #15
    class_descs.append('No entry for trucks') #16
    class_descs.append('One way street') #17
    class_descs.append('Danger')
    class_descs.append('Attention road curves left') #19
    class_descs.append('Attention road curves right') #20
    class_descs.append('Attention S curve')
    class_descs.append('Attention bumpy road') #22
    class_descs.append('Attention slippery road')
    class_descs.append('Attention road will narrow') #24
    class_descs.append('Attention construction site')
    class_descs.append('Attention traffic lights') #26
    class_descs.append('Attention pedestrians')
    class_descs.append('Attention playing children') #28
    class_descs.append('Attention bicycle')
    class_descs.append('Attention snowfall') #30
    class_descs.append('Attention deer crossing')
    class_descs.append('Derestriction') #32
    class_descs.append('Turn right')
    class_descs.append('Turn left') #34
    class_descs.append('Forward')
    class_descs.append('Forward or right') #36
    class_descs.append('Forward or left')
    class_descs.append('Pass right') #38
    class_descs.append('Pass left')
    class_descs.append('Roundabout') #40
    class_descs.append('Derestriction overtaking')
    class_descs.append('Derestriction overtaking for trucks') #4
    for class_idx in classes:
        dump_file = ("gtsrb_%02d.dump" % class_idx)       
        file = open(dataset_path + dump_file, 'rb')
        image_block_class = pickle.load(file)
        file.close()
        img_idxs = np.arange(image_block_class.shape[3])
        np.random.shuffle(img_idxs)
        image_block = np.concatenate( 
            [image_block, image_block_class[:,:,:,img_idxs[0:max_num_imgs_per_class]] ], axis=3)
        labels = np.concatenate( 
            [labels, [class_idx] * len(img_idxs[0:max_num_imgs_per_class])], axis=0 )      
    image_block = np.transpose( image_block, axes=[3, 0, 1, 2]) 
    return image_block, labels, class_descs, sign_ids

# Helper function to split the dataset
def get_train_valid_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def save_dataset(dataset_path, classes, imgs_per_class=10000):
    # Load RGB images
    [imgs, labels, class_descs, sign_ids] = load_gtsrb_images(dataset_path, classes, 10000)  
    imgs = imgs.astype(np.uint8)
    # Transform y labels to one-hot array
    x, y = imgs, labels
    # Shuffle data first
    idx = np.random.randint(0, x.shape[0], x.shape[0])
    x, y = x[idx], y[idx]
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    # Transform x images to vector
    x = np.array([x_k[:].flatten() for x_k in x])
    # Split the dataset
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = get_train_valid_test(x, y)
    # Whiten the data
    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)
    x_train = np.array([(x_k - x_train_mean) / x_train_std for x_k in x_train])
    # Save the data to npy files
    np.save(dataset_path+"x_train.npy", x_train)
    np.save(dataset_path+"y_train.npy", y_train)
    np.save(dataset_path+"x_valid.npy", x_valid)
    np.save(dataset_path+"y_valid.npy", y_valid)
    np.save(dataset_path+"x_test.npy", x_test)
    np.save(dataset_path+"y_test.npy", y_test)

def load_dataset(dataset_path):
    x_train = np.load(dataset_path+"x_train.npy")
    y_train = np.load(dataset_path+"y_train.npy")
    x_valid = np.load(dataset_path+"x_valid.npy")
    y_valid = np.load(dataset_path+"y_valid.npy")
    x_test = np.load(dataset_path+"x_test.npy")
    y_test = np.load(dataset_path+"y_test.npy")
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

class GTSRB:
    x_train, y_train, x_valid, y_valid, x_test, y_test = None, None, None, None, None, None
    train_size, valid_size, test_size = 0, 0, 0
    num_classes, img_width, img_height, img_depth = 0, 0, 0, 0
 
    def __init__(self, dataset_path, num_classes):
        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test) = load_dataset(dataset_path)
        # get params
        self.num_classes = num_classes
        self.img_width = 32
        self.img_height = 32
        self.img_depth = 3
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_width, self.img_height, self.img_depth)
        self.x_valid = self.x_valid.reshape(self.x_valid.shape[0], self.img_width, self.img_height, self.img_depth)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_width, self.img_height, self.img_depth)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.valid_size = self.x_valid.shape[0]
        self.test_size = self.x_test.shape[0]

    def data_augmentation(self, augment_size=5000): 
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=True)
        # fit data for zca whitening
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def next_train_batch(self, batch_size):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx]
        epoch_y = self.y_train[randidx]
        return epoch_x, epoch_y

    def next_valid_batch(self, batch_size):
        randidx = np.random.randint(self.valid_size, size=batch_size)
        epoch_x = self.x_valid[randidx]
        epoch_y = self.y_valid[randidx]
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]