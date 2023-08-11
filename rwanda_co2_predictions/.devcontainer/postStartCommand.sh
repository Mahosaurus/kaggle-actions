sudo update-ca-certificates
pip install poetry --trusted-host pypi.org

export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

poetry install
poetry run pre-commit install --install-hooks
poetry run pre-commit validate-manifest
