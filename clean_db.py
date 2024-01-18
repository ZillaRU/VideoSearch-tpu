import os
import shutil

class DbCleaner():
    def __init__(self):
        self.clean_dir = ['video_collection', 'scene_snapshot', 'dbs']


    def claen_file(self):
        for folder in self.clean_dir:
            try:
                # Delete the entire folder and its contents
                shutil.rmtree(folder)
                print(f"Deleted all files in: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")

        for folder in self.clean_dir:
            if folder != 'dbs':
                try:
                    os.mkdir(folder)
                    print(f"Directory created: {folder}")
                except FileExistsError:
                    print(f"Directory '{folder}' already exists.")
                except Exception as e:
                    print(f"Error creating directory '{folder}': {e}")
            else:
                try:
                    directory_path = os.path.join(folder, 'CH')
                    os.makedirs(directory_path)
                    print(f"Directory and subdirectories created: {directory_path}")
                    directory_path = os.path.join(folder, 'EN')
                    os.makedirs(directory_path)
                    print(f"Directory and subdirectories created: {directory_path}")
                except FileExistsError:
                    print(f"Directory '{directory_path}' or its subdirectories already exist.")
                except Exception as e:
                    print(f"Error creating directory '{directory_path}' and subdirectories: {e}")


            # if folder != 'dbs':
            #     file_list = os.listdir(folder)
            #     # Iterate over each file and delete it
            #     for file_name in file_list:
            #         file_path = os.path.join(folder, file_name)
            #         try:
            #             os.remove(file_path)
            #             print(f"Deleted: {file_path}")
            #         except Exception as e:
            #             print(f"Error deleting {file_path}: {e}")
            # else:
            #     os

a = DbCleaner()
a.claen_file()