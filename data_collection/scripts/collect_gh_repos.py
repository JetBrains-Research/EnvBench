import logging
import multiprocessing
import os
import tarfile

import git
import hydra
import jsonlines
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class RepoDataCollectionConfig(BaseModel):
    num_workers: int
    """Number of processes to run simultaneously.
    (each process will clone a repo and upload to YT)
    """
    input_path: str
    """Path to input data. Input data is expected to be stored in JSONL format and contain 'repo_name' key.
    Will be appended to the path specified in `DATA_ROOT` environment variable."""
    temp_dir: str
    """Path to directory where repositories will be cloned to. 
    Will be appended to the path specified in `DATA_ROOT` environment variable."""
    output_dir: str
    """Path to directory where archived repositories will be saved. 
    Will be appended to the path specified in `DATA_ROOT` environment variable."""


class RepoProcessor:
    def __init__(
        self,
        temp_dir: str,
        output_dir: str,
    ):
        self.temp_dir = temp_dir
        self.output_dir = output_dir

    def _clone_repo(self, repo_name: str) -> str:
        tmp_repo_path = os.path.join(self.temp_dir, repo_name.replace("/", "__"))
        git.Repo.clone_from(f"https://github.com/{repo_name}.git", tmp_repo_path)
        repo = git.Repo(tmp_repo_path)
        repo.remotes.origin.fetch()
        return tmp_repo_path

    def _compress_repo(self, repo_path: str, repo_name: str) -> str:
        final_path = os.path.join(self.output_dir, f"{repo_name.replace('/', '__')}.tar.gz")
        with tarfile.open(final_path, "w:gz") as tar:
            tar.add(
                repo_path,
                arcname=os.path.basename(final_path),
            )

        return final_path

    def __call__(self, repo_name: str) -> None:
        repo_path = self._clone_repo(repo_name=repo_name)
        logging.info("Cloned repo!")
        self._compress_repo(repo_path=repo_path, repo_name=repo_name)
        logging.info("Compressed repo!")


def process_repo(
    temp_dir: str,
    output_dir: str,
    repo_name: str,
) -> None:
    repo_processor = RepoProcessor(temp_dir=temp_dir, output_dir=output_dir)
    return repo_processor(repo_name=repo_name)


@hydra.main(version_base="1.1", config_path="../configs", config_name="collect_gh_repos")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    # TODO: mypy error: Keywords must be strings
    cfg_model = RepoDataCollectionConfig(**cfg)  # type: ignore[misc]

    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    if data_root is None:
        raise ValueError("Please set an environment variable `DATA_ROOT` to specified where data is stored.")

    temp_dir = to_absolute_path(os.path.join(data_root, cfg_model.temp_dir))
    os.makedirs(temp_dir, exist_ok=True)
    output_dir = to_absolute_path(os.path.join(data_root, cfg_model.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    with jsonlines.open(to_absolute_path(os.path.join(data_root, cfg_model.input_path)), "r") as reader:
        repos = [line["repo_name"] for line in reader]

    logging.info(f"Got {len(repos)} repositories to process!")

    with multiprocessing.Pool(processes=cfg_model.num_workers) as pool:
        pool.starmap(
            process_repo,
            ((temp_dir, output_dir, repo) for repo in repos),
        )


if __name__ == "__main__":
    main()
