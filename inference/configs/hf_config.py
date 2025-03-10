from pydantic import BaseModel


class HFConfig(BaseModel):
    upload: bool
    """Should the trajectories be uploaded to HuggingFace or not."""
    repo_id: str
    """Name of the dataset on HuggingFace."""
    path_in_repo: str
    """Path in the dataset to upload the trajectories and configuration to."""
