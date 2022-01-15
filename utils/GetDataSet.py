import os
import urllib.request
import zipfile

class GetDataSet():

    def __init__(self, dataset_link:str, dataset_path:str):
        self._dataset_link= dataset_link
        self._dataset_path = dataset_path


    def download_dataset(self):
        if not os.path.exists(self._dataset_path):
            print('Directory does not exist, New Directory has been Created')
            os.mkdir('Data')

        dataset_url = str(self._dataset_link)
        temp_dataset_path = os.path.join(self._dataset_path, 'dataset_archive.zip')
        print('Downloading Dataset .....')
    
        urllib.request.urlretrieve(dataset_url, temp_dataset_path)
        dataset_dir = os.path.join(self._dataset_path, 'Dataset')
        print('Downloading Done')

        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)


    def extracting(self):
        temp_dataset_path = os.path.join(self._dataset_path, 'dataset_archive.zip')
        dataset_dir = os.path.join(self._dataset_path, 'Dataset')
    
        with zipfile.ZipFile(temp_dataset_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            os.remove(temp_dataset_path)
            print('Extracting Done')

    def run(self):
        self.download_dataset()
        self.extracting()
        return 1


if __name__=="__main__":

    url = 'https://bit.ly/3GuMVI9' #www.kaggle.com/tongpython/cat-and-dog
    folder_name = 'Data'
    Get_Dataset= GetDataSet(url, folder_name)
    Get_Dataset.run()
