from clearml import Dataset,Task
import json

def create_dataset(folder_path, dataset_project, dataset_name):
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        # child_dataset.add_files(folder_path)
        # ipdb.set_trace()
        child_dataset.sync_folder(folder_path)
        child_dataset.upload()
        # child_dataset.finalize()
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        # dataset.add_files(folder_path)
        dataset.sync_folder(folder_path)
        dataset.upload(output_url='s3://public-data/jerex')
        # dataset.finalize()
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])

def main():

    # task = Task.init(project_name="Jerex_DWIE", task_name="delete dataset")
    # Dataset.delete(dataset_id='db3d02093a034822a40a0f8e8a3cd9f8')

    task = Task.init(project_name="PURE", task_name="upload DOCRED",output_uri="s3://ecs.dsta.ai:80/public-data")
    print("creating new dataset!!!")

    dataset = Dataset.create(dataset_name="re3d", dataset_project="datasets/PURE")
    # dataset.add_files(folder_path)
    dataset.add_files("spanNER/data/re3d")
    dataset.upload(output_url='s3://ecs.dsta.ai:80/public-data')
    dataset.finalize()


if __name__ == '__main__':
    main()