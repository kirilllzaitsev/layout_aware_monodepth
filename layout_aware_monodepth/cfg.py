class cfg:
    num_epochs = 10
    vis_freq_epochs = 1
    exp_disabled = False
    do_save_model = True
    line_op = None
    do_overfit = True
    use_single_sample = True

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
