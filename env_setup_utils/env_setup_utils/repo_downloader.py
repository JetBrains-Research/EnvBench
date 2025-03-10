import logging
import os
import shutil
import tarfile
from typing import Literal

import git
from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]


class RepoDownloader:
    def __init__(self, output_dir: str, hf_name: str, language: str):
        self.hf_name = hf_name
        self.output_dir = output_dir
        self.language = language

    def get_repo_archive_path(self, repo_name: str, commit_sha: str, archive_type: Literal["zip", "tar.gz"]):
        return os.path.join(
            self.output_dir,
            f"{self.get_repo_dir_name(repo_name, commit_sha)}.{archive_type}",
        )

    def get_repo_dir_path(self, repo_name: str, commit_sha: str) -> str:
        return os.path.join(self.output_dir, self.get_repo_dir_name(repo_name, commit_sha))

    def get_repo_dir_name(self, repo_name: str, commit_sha: str) -> str:
        return f"{repo_name.replace('/', '__')}@{commit_sha}"

    def _prepare_downloaded_repository(self, repo: git.Repo, commit_sha: str) -> None:
        repo.index.reset(working_tree=True)
        repo.git.clean("-fd")
        repo.head.reset(index=True, working_tree=True)
        repo.git.checkout(commit_sha)
        return None

    def _download_hf(self, repo_name: str, commit_sha: str) -> bool:
        try:
            logging.debug(f"Downloading {repo_name} from Huggingface...")
            download_path = hf_hub_download(
                repo_id=self.hf_name,
                filename=f"repos/{self.language}/{repo_name.replace('/', '__')}.tar.gz",
                repo_type="dataset",
                local_dir=self.output_dir,
            )
        except Exception as e:
            logging.error(f"Failed to download repository '{repo_name}' at commit '{commit_sha}' from HuggingFace.")
            logging.exception(e)
            return False

        local_path = self.get_repo_archive_path(repo_name, commit_sha, "tar.gz")
        try:
            if os.path.exists(local_path):
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path)
                else:
                    os.remove(local_path)
            shutil.move(download_path, local_path)
        except Exception as e:
            logging.error("Failed to move downloaded archive.")
            logging.exception(e)
            return False

        logging.debug(f"Extracting {repo_name}...")
        with tarfile.open(local_path, "r:gz") as tar:
            tar.extractall(path=self.output_dir)

        try:
            if os.path.exists(os.path.join(self.output_dir, os.path.basename(download_path))) and os.path.isdir(
                os.path.join(self.output_dir, os.path.basename(download_path))
            ):
                if os.path.exists(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha)):
                    if os.path.isdir(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha)):
                        shutil.rmtree(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha))
                    else:
                        os.remove(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha))
                os.rename(
                    os.path.join(self.output_dir, os.path.basename(download_path)),
                    self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha),
                )
        except Exception as e:
            logging.error(f"Failed to move downloaded archive to {self.get_repo_dir_name(repo_name, commit_sha)}.")
            logging.exception(e)
            return False

        try:
            logging.debug(f"Checkouting {repo_name} to commit {commit_sha}...")
            repo = git.Repo(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha))
            self._prepare_downloaded_repository(repo=repo, commit_sha=commit_sha)
        except Exception as e:
            logging.error(f"Failed to checkout repository '{repo_name}' to commit '{commit_sha}'.")
            logging.exception(e)
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                shutil.rmtree(os.path.join(self.output_dir, f"repos/{self.language}"))
            except Exception as e:
                logging.warning(f"Couldn't clean remaining files for {repo_name}@{commit_sha} downloaded from HF.")
                logging.exception(e)

            return False
        return True

    def _download_github(self, repo_name: str, commit_sha: str) -> bool:
        try:
            repo = git.Repo.clone_from(
                f"https://github.com/{repo_name}", self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha)
            )
            self._prepare_downloaded_repository(repo=repo, commit_sha=commit_sha)
            return True
        except Exception as e:
            logging.error(f"Failed to download repository '{repo_name}' at commit '{commit_sha}' from GitHub.")
            logging.exception(e)
            return False

    def download(self, repo_name: str, commit_sha: str) -> bool:
        exists = os.path.exists(self.get_repo_dir_path(repo_name, commit_sha))
        if exists:
            try:
                repo = git.Repo(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha))
                self._prepare_downloaded_repository(repo=repo, commit_sha=commit_sha)
                return True
            except Exception as e:
                logging.error(
                    f"Failed to checkout already present repository '{repo_name}' to commit '{commit_sha}' Will try to download again."
                )
                logging.exception(e)
                try:
                    shutil.rmtree(self.get_repo_dir_path(repo_name=repo_name, commit_sha=commit_sha))
                except Exception as e:
                    logging.error(f"Couldn't clean already present repository '{repo_name}.")
                    logging.exception(e)
                    return False

        os.makedirs(self.output_dir, exist_ok=True)

        is_downloaded_from_github = self._download_github(repo_name=repo_name, commit_sha=commit_sha)
        if is_downloaded_from_github:
            return True

        is_downloaded_from_hf = self._download_hf(repo_name=repo_name, commit_sha=commit_sha)
        return is_downloaded_from_hf

    def clear_repo(self, repo_name: str, commit_sha: str):
        for archive_type in ["zip", "tar.gz"]:
            archive_path = self.get_repo_archive_path(
                repo_name=repo_name,
                commit_sha=commit_sha,
                archive_type=archive_type,  # type: ignore[arg-type]
            )
            if os.path.exists(archive_path):
                os.remove(archive_path)

        repo_dir = self.get_repo_dir_path(repo_name, commit_sha)
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
