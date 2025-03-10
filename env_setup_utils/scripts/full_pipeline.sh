#!/usr/bin/env bash
set -e

DATE=$(date '+%Y-%m-%d_%H-%M-%S')

chmod 600 ~/.ssh/id_rsa
sudo chmod 666 /var/run/docker.sock

# install yq
export VERSION=v4.44.3
export BINARY=yq_linux_amd64
wget https://github.com/mikefarah/yq/releases/download/${VERSION}/${BINARY}.tar.gz -O - |\
  tar xz && sudo mv ${BINARY} /usr/bin/yq

# Get run name
export PATH_IN_REPO="${LANGUAGE}_${CLIENT}_${DATE}"
echo "Running full inference and evaluation for $PATH_IN_REPO"

# client configuration
CONFIG_FILE="inference/configs/$CONFIG_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file $CONFIG_FILE not found"
  exit 1
fi
# modify config file using yq
yq -i ".agent.instruction_provider.language = \"$LANGUAGE\"" $CONFIG_FILE
yq -i ".data_source.hf.split = \"$LANGUAGE\"" $CONFIG_FILE
yq -i ".docker.image = \"$IMAGE\"" $CONFIG_FILE
yq -i ".client = \"$CLIENT\"" $CONFIG_FILE
yq -i ".hf.path_in_repo = \"$PATH_IN_REPO\"" $CONFIG_FILE
yq -i ".hf.upload = true" $CONFIG_FILE
yq -i ".max_concurrent = $MAX_CONCURRENCY" $CONFIG_FILE

echo "Config file $CONFIG_FILE modified"
cat $CONFIG_FILE

# 1. Run inference
pushd inference
poetry install
poetry run python run_inference.py --config-name "$CONFIG_NAME"
popd

# 2. Process trajectories to scripts
echo "Trajectories collected. Proceeding with scripts.jsonl..."
pushd env_setup_utils
poetry install
poetry run python env_setup_utils/process_trajectories_to_scripts.py --input-trajectories-dir "$PATH_IN_REPO"
popd

# 3. Evaluation
echo "Trajectories converted to scripts. Proceeding with evaluation..."
pushd qodana-eval
poetry install
python main.py output.hf.path_in_repo="$PATH_IN_REPO" input.hf.path_in_repo="$PATH_IN_REPO"/scripts.jsonl operation.pool_config.max_workers=$MAX_WORKERS language=$LANGUAGE
popd

echo "Run $PATH_IN_REPO was executed successfully"
