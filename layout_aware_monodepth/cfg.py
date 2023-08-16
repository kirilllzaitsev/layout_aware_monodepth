import os


class cfg:
    @classmethod
    def params(cls):
        attrs_to_exclude = [
            "path_to_project_dir",
            "base_kitti_dataset_dir",
        ]
        return {
            **{
                k: v
                for k, v in vars(cls).items()
                if not k.startswith("__")
                and not callable(v)
                and not isinstance(v, classmethod)
                and k not in attrs_to_exclude
            }
        }

    is_cluster = os.path.exists("/cluster")
    exp_base_dir = "/cluster/scratch/kzaitse" if is_cluster else "/tmp"
