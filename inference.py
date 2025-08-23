"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import pyvips
import SimpleITK
import numpy
import random
import pandas as pd
from autogluon.tabular import TabularPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "chimera-clinical-data-of-bladder-cancer-patients",
            "tissue-mask",
        ): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input - use thumbnail loading for tissue mask to avoid memory issues with large WSI tissue masks
    input_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/tissue-mask",
        max_size=1024,
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_bladder_cancer_tissue_biopsy_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi",
        max_size=1024,
    )
    input_chimera_clinical_data_of_bladder_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-patients.json",
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Debug: Print information about loaded data
    print("=+=" * 10)
    print("Data Loading Summary:")
    # Handle PyVips objects for tissue mask
    if hasattr(input_tissue_mask, 'width'):
        print(f"Tissue mask size: {input_tissue_mask.width}x{input_tissue_mask.height}")
    else:
        print(f"Tissue mask shape: {input_tissue_mask.shape}")
    
    print(f"Pathology WSI type: {type(input_bladder_cancer_tissue_biopsy_whole_slide_image)}")
    if hasattr(input_bladder_cancer_tissue_biopsy_whole_slide_image, 'width'):
        print(f"Pathology WSI size: {input_bladder_cancer_tissue_biopsy_whole_slide_image.width}x{input_bladder_cancer_tissue_biopsy_whole_slide_image.height}")
    print(f"Clinical data keys: {list(input_chimera_clinical_data_of_bladder_cancer_patients.keys()) if input_chimera_clinical_data_of_bladder_cancer_patients else 'None'}")
    print("=+=" * 10)

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # Try to load a trained AutoGluon model and predict BRS3 probability
    try:
        # Prefer resources root if it looks like an AutoGluon predictor dir
        base_resources_dir = Path("/opt/app/resources")
        models_root = base_resources_dir / "models"

        def _is_predictor_dir(path: Path) -> bool:
            return (path / "predictor.pkl").exists() or (path / "learner.pkl").exists()

        # First, try to use saved across-fold ensemble weights if available
        ensemble_weights_path = models_root / "ensemble_CAT_RF_20.json"
        used_ensemble = False

        candidate_dirs = []
        # 1) resources/
        candidate_dirs.append(base_resources_dir)
        # 2) resources/models/rf_final
        candidate_dirs.append(base_resources_dir / "models" / "rf_final")
        # 3) any subdir under resources/models that contains predictor artifacts
        models_root = base_resources_dir / "models"
        if models_root.exists():
            for sub in models_root.iterdir():
                if sub.is_dir():
                    candidate_dirs.append(sub)

        if ensemble_weights_path.exists():
            try:
                weights = json.loads(ensemble_weights_path.read_text())
                if weights.get('ensemble_name') == 'Ensemble_CAT_RF_20':
                    # Build clinical DF
                    clinical_df = pd.DataFrame([input_chimera_clinical_data_of_bladder_cancer_patients])
                    label_col = weights.get('label', 'BRS_binary')
                    pos_class = weights.get('positive_class', 'BRS3')
                    if label_col in clinical_df.columns:
                        clinical_df = clinical_df.drop(columns=[label_col])

                    fold_rel_paths = weights.get('fold_rel_paths', [])
                    fam_models = weights.get('family_model_by_fold', {})
                    fam_weights = weights.get('weights', {'CAT': 0.5, 'RF': 0.5})

                    fam_preds = { 'CAT': [], 'RF': [] }

                    # Optional dependency checks (best-effort)
                    try:
                        import xgboost  # noqa: F401
                    except Exception:
                        pass
                    try:
                        import catboost  # noqa: F401
                    except Exception:
                        pass
                    try:
                        import lightgbm  # noqa: F401
                    except Exception:
                        pass

                    for i, rel in enumerate(fold_rel_paths, start=1):
                        fold_dir = base_resources_dir / rel
                        try:
                            p = TabularPredictor.load(str(fold_dir))
                        except Exception as _e:
                            print(f"Warn: failed to load fold predictor {fold_dir}: {_e}")
                            continue
                        chosen = fam_models.get(str(i), {})
                        # Ensure we skip ensembles
                        model_names = [m for m in p.model_names() if not m.startswith('WeightedEnsemble')]
                        for fam in ['CAT', 'RF']:
                            mname = chosen.get(fam)
                            if not mname:
                                # fallback: try to detect
                                patterns = {'CAT': ['CatBoost','CAT'], 'RF': ['RandomForest','RF']}
                                upnames = {m: m.upper() for m in model_names}
                                for m in model_names:
                                    for pat in patterns[fam]:
                                        if pat.upper() in upnames[m]:
                                            mname = m
                                            break
                                    if mname:
                                        break
                            if not mname:
                                continue
                            try:
                                proba = p.predict_proba(clinical_df, model=mname)
                                if hasattr(proba, 'columns'):
                                    col = pos_class if pos_class in proba.columns else proba.columns[-1]
                                    fam_preds[fam].append(float(proba[col].iloc[0]))
                                else:
                                    fam_preds[fam].append(float(proba[0]))
                            except Exception as _e:
                                print(f"Warn: prediction failed for fold {i}, family {fam}: {_e}")

                    if fam_preds['CAT'] and fam_preds['RF']:
                        cat_avg = float(sum(fam_preds['CAT']) / len(fam_preds['CAT']))
                        rf_avg  = float(sum(fam_preds['RF'])  / len(fam_preds['RF']))
                        output_brs_binary_classification = float(
                            fam_weights.get('CAT', 0.5) * cat_avg + fam_weights.get('RF', 0.5) * rf_avg
                        )
                        print(
                            f"Ensemble_CAT_RF_20 prediction: CAT_avg={cat_avg:.4f}, RF_avg={rf_avg:.4f}, "
                            f"weights(CAT={fam_weights.get('CAT', 0.5):.3f}, RF={fam_weights.get('RF', 0.5):.3f}) => "
                            f"prob BRS3={output_brs_binary_classification:.4f}"
                        )
                        used_ensemble = True
            except Exception as ens_err:
                print(f"Warning: failed to use Ensemble_CAT_RF_20 weights: {ens_err}")

        if not used_ensemble:
            model_dir = None
            for cand in candidate_dirs:
                try:
                    if cand.exists() and _is_predictor_dir(cand):
                        model_dir = cand
                        break
                except Exception:
                    continue

            if model_dir is None:
                raise FileNotFoundError(
                    f"No valid AutoGluon predictor directory found under {base_resources_dir}"
                )

            print(f"Loading AutoGluon predictor from: {model_dir}")
            # Ensure XGBoost is available if the predictor depends on it
            try:
                import xgboost  # type: ignore  # noqa: F401
            except Exception as dep_err:
                raise RuntimeError(
                    "Required dependency 'xgboost' is not installed. "
                    "Add 'xgboost' to requirements.txt, rebuild the image, and retry."
                ) from dep_err
            try:
                predictor = TabularPredictor.load(str(model_dir))
            except Exception as load_err:
                # Handle Python version mismatch gracefully
                if "Python version" in str(load_err) or "require_py_version_match" in str(load_err):
                    print(
                        "Warning while loading predictor (likely Python version mismatch). "
                        "Retrying with require_py_version_match=False."
                    )
                    predictor = TabularPredictor.load(
                        str(model_dir), require_py_version_match=False
                    )
                else:
                    # Bubble up dependency-related hints if applicable
                    if "No module named 'xgboost'" in str(load_err) or "No module named xgboost" in str(load_err):
                        raise RuntimeError(
                            "AutoGluon predictor requires 'xgboost' but it was not found at runtime. "
                            "Ensure 'xgboost' is listed in requirements.txt and rebuild the Docker image."
                        ) from load_err
                    raise

            # Convert clinical JSON to DataFrame
            clinical_df = pd.DataFrame([input_chimera_clinical_data_of_bladder_cancer_patients])

            # Ensure target column is not present
            if predictor.label in clinical_df.columns:
                clinical_df = clinical_df.drop(columns=[predictor.label])

            # Predict probabilities and extract BRS3 probability
            proba_df = predictor.predict_proba(clinical_df)

            # Determine positive class
            positive_class = None
            # Prefer model-local metadata, then fall back to base resources metadata
            metadata_paths = [model_dir / "model_metadata.json", base_resources_dir / "model_metadata.json"]
            for metadata_path in metadata_paths:
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text())
                        positive_class = metadata.get("positive_class")
                        if positive_class:
                            break
                    except Exception:
                        pass

            target_class = positive_class or "BRS3"

            # Handle both DataFrame (multiclass) and Series/ndarray (binary) cases
            if hasattr(proba_df, 'columns'):
                # DataFrame case
                if target_class in proba_df.columns:
                    output_brs_binary_classification = float(proba_df[target_class].iloc[0])
                else:
                    output_brs_binary_classification = float(proba_df.max(axis=1).iloc[0])
            else:
                # Series / ndarray case (assumed positive class probability)
                output_brs_binary_classification = float(proba_df[0])

            print(f"Model prediction (prob {target_class}): {output_brs_binary_classification:.4f}")
    except Exception as e:
        # Fail fast and loudly; do not emit random predictions
        raise

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "brs-probability.json",
        content=output_brs_binary_classification,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    """
    Load image files using appropriate library based on file type:
    - PyVips for pathology images: .tif, .tiff, .mrxs, .svs, .ndpi
    - SimpleITK for radiology images: .mha
    """
    # Find all compatible files
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.mha':
        # Use SimpleITK for radiology images (.mha)
        print(f"Loading radiology image using SimpleITK: {file_path}")
        image = SimpleITK.ReadImage(file_path)
        array = SimpleITK.GetArrayFromImage(image)
        return array
    
    else:
        # Use PyVips for pathology images (.tif, .tiff, .mrxs, .svs, .ndpi)
        print(f"Loading pathology image using PyVips: {file_path}")
        image = pyvips.Image.new_from_file(file_path)
        
        # For very large images, you might want to downsample first
        # Uncomment the next line to downsample by factor of 4 to save memory
        # image = image.resize(0.25)
        
        # Convert to numpy array
        # Note: This will load the entire image into memory
        # For production use, consider processing tiles instead
        memory_image = image.write_to_memory()
        array = numpy.frombuffer(memory_image, dtype=numpy.uint8)
        
        # Reshape based on image dimensions and bands
        height = image.height
        width = image.width
        bands = image.bands
        
        if bands == 1:
            # Grayscale image
            array = array.reshape((height, width))
        else:
            # Multi-channel image (RGB, RGBA, etc.)
            array = array.reshape((height, width, bands))
        
        return array


def load_image_file_as_thumbnail(*, location, max_size=1024):
    """
    Load image as a thumbnail for memory-efficient processing of WSIs
    This is recommended for actual whole slide images
    Returns the PyVips image object directly for memory efficiency
    """
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    print(f"Loading pathology image as thumbnail using PyVips: {file_path}")
    
    # Load image with PyVips
    image = pyvips.Image.new_from_file(file_path)
    
    # Calculate downsampling factor to fit within max_size
    scale_factor = min(max_size / image.width, max_size / image.height)
    if scale_factor < 1.0:
        print(f"Downsampling image by factor {scale_factor:.3f} (from {image.width}x{image.height} to {int(image.width*scale_factor)}x{int(image.height*scale_factor)})")
        image = image.resize(scale_factor)
    else:
        print(f"Image size {image.width}x{image.height} is within max_size={max_size}, no downsampling needed")
    
    # Return the PyVips image object directly (much more memory efficient)
    return image


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())