FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PyVips
RUN apt-get update && apt-get install -y \
    libvips-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Normalize AutoGluon predictor metadata to the container's Python version
# so runtime won't warn about version mismatches.
RUN python - <<'PY'
from pathlib import Path
from autogluon.tabular import TabularPredictor

resources_dir = Path('/opt/app/resources')
try:
    if (resources_dir / 'predictor.pkl').exists() or (resources_dir / 'learner.pkl').exists():
        pred = TabularPredictor.load(str(resources_dir), require_py_version_match=False)
        # Re-save in-place to update stored metadata to current Python version
        pred.save(str(resources_dir))
        print('Re-saved AutoGluon predictor to match container Python version.')
    else:
        print('No root AutoGluon predictor found; skipping normalization step.')
except Exception as e:
    print(f'Skipping predictor normalization due to error: {e}')
PY

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
