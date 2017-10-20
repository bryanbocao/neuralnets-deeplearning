# Reference: https://github.com/michael-iuzzolino/CIFAR_reader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.ndimage
import tarfile
from sklearn.utils import shuffle
from six.moves import urllib
try:
    import cPickle as pickle
    PY_VERSION = 2
except ImportError:
    import pickle
    PY_VERSION = 3

class CIFAR_reader(object):

    def __init__(self, one_hot=True, verbose=True, img_size=32, num_classes=10, augment=False):
        """
            Pieces of code taken from:
                https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/download.py
        """
        self._one_hot_encoding = one_hot
        self._verbose = verbose

        self._train_files = ["data_batch_{}".format(i+1) for i in range(5)]
        self._test_file = ["test_batch"]

        self._CIFAR_data = { key : {"data" : None, "labels" : None} for key in ["training", "test"]}
        self._label_names = None
        self._num_classes = num_classes
        self._img_size = img_size
        self._augment = augment

        self._new_batch = True
        self._batch_i = 0
        self._num_batches = None


        # DOWNLOAD PARAMS
        # ----------------------------------------------------------------------------
        DOWNLOAD_DIR = os.path.join("data", "CIFAR-10")
        SOURCE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self._download_params = {
            "download_found" : False,
            "download_dir"   : DOWNLOAD_DIR,
            "source_url"     : SOURCE_URL,
            "filename"       : SOURCE_URL.split('/')[-1],
            "filepath"       : os.path.join(DOWNLOAD_DIR, SOURCE_URL.split('{}'.format(os.path.sep))[-1])
        }
        # ----------------------------------------------------------------------------

        # DATA AUGMENTATION
        # ----------------------------------------------------------------------------
        self._augmentation_dir = os.path.join(DOWNLOAD_DIR, "augmented_data")
        self._augmentation_filename = "augmented_data"
        self._augmentation_filepath = os.path.join(self._augmentation_dir, self._augmentation_filename)
        # ----------------------------------------------------------------------------

        # Init checking and download / extraction if necessary
        self._find_CIFAR_data()

        self._unpack()

    def _find_CIFAR_data(self):
        """
        Checks directory for CIFAR data.
        :param:
            Nothing.
        :return:
            Nothing.
        """

        print("Checking for CIFAR data...")

        # Check if the file already exists.
        # If D.N.E., download and extract,
        # Else if it exists, assume it has been extracted already
        if not os.path.exists(self._download_params["filepath"]):

            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(self._download_params["download_dir"]):
                os.makedirs(self._download_params["download_dir"])

            self._download_CIFAR_data()

        self._extract_CIFAR_data()

    def _download_CIFAR_data(self):
        """
        Download and extract the data if it doesn't already exist.
        Assumes the url is a tar-ball file.
        :param url:
            Internet URL for the tar-file to download.
            Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        :param download_dir:
            Directory where the downloaded file is saved.
            Example: "data/CIFAR-10/"
        :return:
            Nothing.
        """

        # Retrieve the data
        filepath, _ = urllib.request.urlretrieve(url=self._download_params["source_url"],
                                                  filename=self._download_params["filepath"],
                                                  reporthook=self._print_download_progress)

        print("\n")

    def _extract_CIFAR_data(self):
        """
        Extract data from CIFAR data download
        :params:
            Nothing.
        :return:
            Nothing.
        """
        print("Extracting Data...")

        filename = self._download_params["filepath"]
        extract_path = self._download_params["download_dir"]

        tar_open_flag = "r:gz" if filename.endswith("tar.gz") else "r:"
        tar = tarfile.open(filename, tar_open_flag)

        # Check for possible download error - in which case, redownload file
        try:
            tar.extractall(path=extract_path)
        except (IOError, EOFError):
            print("** Error: Extraction Failure. Redownloading data...")
            self._download_CIFAR_data()
            self._extract_CIFAR_data()

        self._download_params["extract_dir"] = os.path.join(extract_path, tar.getnames()[0])
        tar.close()

    def _print_download_progress(self, count, block_size, total_size):
        """
        Function used for printing the download progress.
        Used as a call-back function in maybe_download_and_extract().
        """

        # Percentage completion.
        pct_complete = float(count * block_size) / total_size

        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    def _load_data(self, file_list, data_set):
        """
        Read the data from the extraction and load it into data dictionary
        :param file_list:
            The list of CIFAR files to read from
            Example: ["data_batch_1", "data_batch_2", ... , "data_batch_5"]
        :param data_set:
            Name of the dataset we are processing
            Example: "training" or "test"
        :return:
            Nothing. Stores all processing to class dict object, self._CIFAR_data
        """
        extraction_dir = self._download_params["extract_dir"]
        for i, file_name in enumerate(file_list):
            filepath = os.path.join(extraction_dir, file_name)

            with open(filepath, 'rb') as infile:
                train_dict = pickle.load(infile)

                if self._verbose:
                    print("Loading {}...".format(train_dict["batch_label"]))

                data = train_dict["data"]
                data = data.reshape(len(data), 3, self._img_size, self._img_size).transpose(0, 2, 3, 1).astype("uint8")
                labels = train_dict["labels"]

                if self._one_hot_encoding:
                    one_hot_labels = np.zeros((len(labels), self._num_classes))
                    for i, label in enumerate(labels):
                        one_hot_labels[i][label] = 1
                    labels = one_hot_labels

                if self._CIFAR_data[data_set]["data"] is None:
                    self._CIFAR_data[data_set]["data"] = data
                    self._CIFAR_data[data_set]["labels"] = labels
                else:
                    self._CIFAR_data[data_set]["data"] = np.r_[self._CIFAR_data[data_set]["data"], data]
                    self._CIFAR_data[data_set]["labels"] = np.r_[self._CIFAR_data[data_set]["labels"], labels]

    def _load_augmented_data(self):
        print("Loading augmented data from {}...".format(self._augmentation_filepath))
        with open(self._augmentation_filepath, 'rb') as infile:
            print(infile)
            augmented_data = pickle.load(infile)
        print("Loading augmented data complete.")

        self._CIFAR_data["training"]["data"] = augmented_data["data"]
        self._CIFAR_data["training"]["labels"] = augmented_data["labels"]

    def _write_augmented_data(self, data):
        print("Writing augmented data to {}...".format(self._augmentation_filepath))
        with open(self._augmentation_filepath,'wb') as outfile:
            pickle.dump(data, outfile)
        print("Writing augmented data complete.")

    def _augment_data(self, rotate_data=False, flip_data=True):
        """
        Augment the dataset.
        See the following SE for ideas:
            https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        :param :
        :return:
            Nothing.
        """

        if not os.path.exists(self._augmentation_filepath):
            # Check if the download directory exists, otherwise create it.
            if not os.path.exists(self._augmentation_dir):
                os.makedirs(self._augmentation_dir)

        print("Augmenting Data...")

        X_augmented = []
        y_augmented = []
        for x, y in zip(self._CIFAR_data["training"]["data"], self._CIFAR_data["training"]["labels"]):

            X_augmented.append(x)
            y_augmented.append(y)

            # Rotation
            if rotate_data:
                rotation_1 = scipy.ndimage.interpolation.rotate(x, 90)
                rotation_2 = scipy.ndimage.interpolation.rotate(x, 270)

                X_augmented.append(rotation_1)
                X_augmented.append(rotation_2)
                y_augmented.append(y)
                y_augmented.append(y)

            # TODO: Translation

            # TODO: Rescaling

            # Flipping
            if flip_data:
                left_right_flip = np.fliplr(x)
                X_augmented.append(left_right_flip)
                y_augmented.append(y)

                # NOTE: Commented out for testing - produces very large dataset that takes considerable time to run and save
                # Feel free to optimize this
                # ------------------------------------------------
                # inverted_flip = np.flip(x, axis=0)
                # X_augmented.append(inverted_flip)
                # y_augmented.append(y)
                #
                # l_r_inverted_flip = np.fliplr(inverted_flip)
                # X_augmented.append(l_r_inverted_flip)
                # y_augmented.append(y)
                # ------------------------------------------------

            # TODO: Shearing

            # TODO: Stretching

        # Store augmented data
        self._write_augmented_data({"data" : np.array(X_augmented), "labels" : np.array(y_augmented)})

        self._CIFAR_data["training"]["data"] = np.array(X_augmented)
        self._CIFAR_data["training"]["labels"] = np.array(y_augmented)

    def _data_stats(self):
        """
        Calculate the basic stats of the dataset.
        :param:
            Nothing.
        :return:
            Nothing. Populates training, test, classes, counts, distribution of classes, etc.
        """
        num_training = self._CIFAR_data["training"]["data"].shape[0]
        num_test = self._CIFAR_data["test"]["data"].shape[0]

        self.__dict__.update({"num_training_examples" : num_training})
        self.__dict__.update({"num_test_examples" : num_test})

    def _unpack(self):
        """
        Unpack the data from the extracted files
        :param augment:
            Bool indicating desire to apply data augmentation
        :return:
            Dict, self._CIFAR_data, containing training / test X and y in addition to meta data (class labels)
        """
        print("Unpacking data...")

        # Check if augmented data flag is ON
        # If yes, check if data already augmented
        #   If already augmented, load
        #   Else, load training data and augment it
        # Else, load vanilla data
        if self._augment:
            if os.path.exists(self._augmentation_filepath):
                self._load_augmented_data()
            else:
                self._load_data(self._train_files, "training")
                self._augment_data()
        else:
            self._load_data(self._train_files, "training")

        # Load testing data
        self._load_data(self._test_file, "test")

        # Populate data stats
        self._data_stats()

        # Meta data
        filepath = os.path.join(self._download_params["extract_dir"], "batches.meta")
        with open(filepath, 'rb') as infile:
            meta_dict = pickle.load(infile)
            self.labels = meta_dict["label_names"]

        return self._CIFAR_data

    @property
    def train(self):
        return self._CIFAR_data["training"]

    @property
    def test(self):
        return self._CIFAR_data["test"]


    def preview_data(self, data_set="training"):
        """
        Previews a randomly chosen image from a dataset
        :param data_set:
            Specifies which dataset - "training" or "test" - to randomly pull from
        :return:
            Nothing. Plots the image with matplotlib.
        """
        random_index = np.random.randint(self._CIFAR_data[data_set]["labels"].shape[0])
        random_img = self._CIFAR_data[data_set]["data"][random_index]

        label = self.labels[np.argmax(self._CIFAR_data[data_set]["labels"][random_index])]

        plt.imshow(random_img)
        plt.title("Class: {}".format(label.capitalize()))
        plt.axis('off')
        plt.show()

    def next_batch(self, batch_size=32):
        """
        Shuffles the dataset and retrieves a batch of size batch_size
        :param batch_size:
            The size of the batch
            Example: 100
        :return:
            X_batch, y_batch
        """

        if self._batch_i > self._num_batches:
            self._new_batch = True

        if self._new_batch:
            self._batch_i = 0
            self._num_batches = (self.num_training_examples / batch_size)
            self._X_shuff, self._y_shuff = shuffle(self._CIFAR_data["training"]["data"], self._CIFAR_data["training"]["labels"])
            self._new_batch = False

        X_batch = self._X_shuff[self._batch_i:self._batch_i+batch_size]
        y_batch = self._y_shuff[self._batch_i:self._batch_i+batch_size]
        self._batch_i += 1

        return X_batch, y_batch

def main():
    cifar = CIFAR_reader(verbose=False)
    print(cifar.labels)
    # print(cifar.train)
    # print(cifar.test)
    print(cifar.num_training_examples)
    print(cifar.num_test_examples)

    cifar.preview_data()

    X_batch, y_batch = cifar.next_batch(100)
    print("X_batch shape: {}".format(X_batch.shape))
    print("y_batch shape: {}".format(y_batch.shape))

if __name__ == "__main__":
    main()
