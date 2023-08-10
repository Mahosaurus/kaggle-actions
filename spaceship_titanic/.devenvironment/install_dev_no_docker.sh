pip install "poetry==1.3.2"
poetry install

poetry run pre-commit install --install-hooks
poetry run pre-commit validate-manifest

bash
