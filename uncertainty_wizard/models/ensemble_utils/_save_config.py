import logging
import os
import shutil


def _preprocess_path(path: str) -> str:
    """
    Preprocesses the path, i.e.,
    it makes sure there's no tailing '/' or '\'
    :param path: The path as passed by the user
    :return: The path of the folder on which the model will be stored
    """
    if path.endswith("/") or path.endswith("\\"):
        path = path[: len(path) - 1]
    return path


class SaveConfig:
    """
    This is a class containing information and utility methods about the saving of the ensemble.
    Currently, it only contains a few static fields. This may change in the future.

    Instances of the SaveConfig can be used in the save_single_model and load_single_model
     methods of EnsembleContextManagers. However, consider SaveConfigs as read-only classes.
     In addition, instances should only be created internally by uncertainty wizard
     and not in custom user code.

    """

    def __init__(
        self, ensemble_save_base_path: str, delete_existing: bool, expect_model: bool
    ):
        """
        ** WARNING ** Typically, there's no need for uwiz users to call this contructor:
        They are created by uncertainty wizard and passed to (potentially custom made) context managers.

        Creates a new save config.
        Note that this automatically triggers the preparation of the specified save_path
        (e.g. the deletion of existing
        :param ensemble_save_base_path: Where to store the ensemble
        :param delete_existing: If the folder is non empty and delete_existing is True,
         the folder will be cleared. If this is false and the folder is non-empty,
         a warning will be logged.
        :param expect_model: If this is True, we expect that there is already an
         ensemble file in the folder and the warning described for 'delete_existing'
         will not be logged.
        """
        self._ensemble_save_base_path = _preprocess_path(ensemble_save_base_path)
        self._create_or_empty_folder(
            path=ensemble_save_base_path,
            overwrite=delete_existing,
            expect_model=expect_model,
        )

    def filepath(self, model_id: int):
        """
        This methods builds the path on which a particular atomic model should be saved / found
        :param model_id: the id of the atomic model
        :return: A path, where a folder named after the model id is appended to self.ensemble_save_base_path
        """
        return f"{self._ensemble_save_base_path}/{model_id}"

    @property
    def ensemble_save_base_path(self) -> str:
        """
        Returns the path (as string) where this ensemble is stored.
        This path is always a folder and after successful creation of the ensemble,
        it will contain the ensemble config file and a subfolder for every atomic model.
        :return: The save path of this ensemble as string
        """
        return self._ensemble_save_base_path

    @staticmethod
    def _create_or_empty_folder(path: str, overwrite: bool, expect_model=False) -> None:
        if os.path.exists(path=path):
            if overwrite:
                shutil.rmtree(path, ignore_errors=False, onerror=None)
                os.mkdir(path)
            elif not expect_model:
                logging.info(
                    f"A folder {path} already exists. "
                    f"We will assume it contains a lazy ensemble. "
                    f"Specify `delete_existing=True` to empty the folder when creating the LazyEnsemble."
                )
        else:
            os.makedirs(path)
