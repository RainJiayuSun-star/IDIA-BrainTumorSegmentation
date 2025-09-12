import sys
import os

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(current_dir)
print(parent_dir)
from submodules.theta_utils.logger import setup_logger
import logging
from collections import defaultdict

# Set up the logger
setup_logger(log_level='DEBUG')

logging.info('🚀 Starting data preparation process...')
logging.debug(f'🐍 Python version: {sys.version}')
logging.debug(f'📁 Current working directory: {os.getcwd()}')

import yaml
import json
import glob
import nibabel as nib
import numpy as np
from pprint import pformat
from random import randint

def load_config(config_path):
    logging.info('📂 Loading configuration file...')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info('✅ Configuration loaded successfully.')
    return config


def check_and_fix_affine_matrices(image_files):
    logging.info('🔍 Checking affine matrices...')
    affine_matrices = []
    for image_file in image_files:
        logging.debug(f'🔍 Loading metadata for {image_file}...')
        img = nib.load(image_file)
        affine_matrices.append(img.affine)

    all_affines_equal = all(np.array_equal(affine_matrices[0], affine) for affine in affine_matrices[1:])
    if all_affines_equal:
        logging.info('✅ All affine matrices are equal.')
        return True, 'equal'

    abs_difference_sum = sum(np.abs(affine_matrices[0] - affine).sum() for affine in affine_matrices[1:])
    if abs_difference_sum < 1e-3:
        logging.info(f'🔄 Affine matrices are slightly different, with {abs_difference_sum = }. Fixing...')
        logging.debug(f'🧮 Original affine matrices: {affine_matrices}')
        mean_affine = np.mean(affine_matrices, axis=0)
        logging.debug(f'🧮 Calculated mean affine matrix: {mean_affine}')
        for image_file in image_files:
            logging.info(f'🔧 Fixing affine matrix for {image_file}...')
            img = nib.load(image_file)
            logging.debug(f'📊 Original image data shape: {img.get_fdata().shape}')
            new_img = nib.Nifti1Image(img.get_fdata(), mean_affine)
            logging.debug(f'📊 New image data shape: {new_img.get_fdata().shape}')
            nib.save(new_img, image_file)
            logging.info(f'💾 Saved fixed image: {image_file}')
        logging.info('✅ All affine matrices fixed.')
        logging.debug(f'🧮 New affine matrix for all images: {mean_affine}')
        return True, 'fixed'
    
    logging.warning('⚠️ Affine matrices are too different. Cannot fix.')
    return False, 'skipped'


def generate_json_for_dataset(data_root, dataset_name, dataset_config, num_folds, image_pattern_keys):
    logging.info(f'🔍 Generating JSON for {dataset_name} dataset...')
    base_path = os.path.join(data_root, dataset_config['base_path'])
    patient_pattern = dataset_config['patient_pattern']
    label_pattern = dataset_config['label_pattern']
    image_patterns = dataset_config['image_patterns']

    logging.debug(f'🔢 Using {num_folds} folds for cross-validation')

    data = {'training': []}
    missing_data = defaultdict(list)
    affine_issues = {'equal': 0, 'fixed': 0, 'skipped': 0}

    logging.info(f'📁 Searching for patient directories in {base_path}...')
    glob_pattern = os.path.join(base_path, patient_pattern)
    logging.debug(f'🔍 Using glob pattern: {glob_pattern = }')
    patient_dirs = glob.glob(glob_pattern)
    logging.debug(f'📁 Number of directories found: {len(patient_dirs)}')

    for patient_dir in patient_dirs:
        logging.debug(f'📁 {patient_dir = }')

        patient = os.path.basename(os.path.basename(patient_dir))  # Get the patient number
        logging.debug(f'👤 {patient = }')

        glob_pattern = os.path.join(patient_dir, label_pattern.format(patient=patient))
        logging.debug(f'🔍 Searching for label file: {glob_pattern = }')
        label_file = glob.glob(glob_pattern)
        if not label_file:
            warning_message = f'⚠️ No label file found for patient {patient}. Skipping this patient! Looking for {glob_pattern}'
            logging.warning(warning_message)
            missing_data['label'].append(patient)
            continue

        image_files = []
        missing_modalities = []
        for modality in image_pattern_keys:  # Use the consistent order of keys
            pattern = image_patterns.get(modality)
            if not pattern:
                logging.warning(f'⚠️ No pattern found for modality {modality} in dataset {dataset_name}. Skipping this modality.')
                missing_modalities.append(modality)
                continue
            logging.debug(f'🔍 Searching for {modality} image with pattern: {pattern.format(patient=patient)}')
            image_file = glob.glob(os.path.join(patient_dir, pattern.format(patient=patient)))
            if not image_file:
                warning_message = f'⚠️ No {modality} image found for patient {patient}. Skipping this patient!'
                logging.warning(warning_message)
                missing_modalities.append(modality)
            else:
                image_files.append(image_file[0])

        if missing_modalities:
            missing_data['modalities'].append((patient, missing_modalities))
            logging.warning(f'⚠️ Patient {patient} has missing modalities. Skipping this patient!')
            continue

        # Check and fix affine matrices
        all_image_files = [label_file[0]] + image_files
        affine_check_result, affine_status = check_and_fix_affine_matrices(all_image_files)
        affine_issues[affine_status] += 1

        if not affine_check_result:
            logging.warning(f'⚠️ Skipping patient {patient} due to affine matrix issues.')
            continue

        logging.debug(f'🔍 Calculating relative paths...')
        relative_label_path = os.path.relpath(label_file[0], start=base_path)
        relative_image_paths = [os.path.relpath(f, start=base_path) for f in image_files]
        logging.debug(f'📄 Relative label path: {relative_label_path}')
        logging.debug(f'🖼️ Relative image paths: {relative_image_paths}')

        # Randomly assign fold
        fold = randint(0, num_folds - 1)
        logging.debug(f'🎲 Assigned fold {fold} to patient {patient}')

        data['training'].append({
            'fold': fold,
            'label': relative_label_path,
            'image': relative_image_paths
        })
        logging.info('✅ Added data entry to training set.')
        logging.debug(f'✅ Added data for patient {patient} with fold {fold}')

    logging.debug(f'🔍 Number of patients found: {len(data["training"])}')
    logging.info(f'✅ JSON generation complete for {dataset_name} dataset.')
    logging.info(f'📊 Affine matrix issues: Equal: {affine_issues["equal"]}, Fixed: {affine_issues["fixed"]}, Skipped: {affine_issues["skipped"]}')
    return data, missing_data, affine_issues


def load_and_prepare_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    logging.info('🚀 Starting JSON generation process...')
    config = load_config(os.path.join(script_dir, 'datasets.yaml'))
    logging.debug(f'⚙️ {config = }')

    num_folds = config.get('folds', 1)  # Get folds from root level, default to 1
    logging.info(f'🔢 Number of folds: {num_folds}')

    logging.debug('🔍 Determining output directory...')
    output_dir = os.path.join(script_dir, 'patients')
    logging.debug(f'📁 {output_dir = }')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'📁 Created directory: {output_dir}')

    data_root = os.path.abspath(os.path.expanduser(config['data_root']))
    logging.debug(f'📁 {data_root = }')

    logging.info('🔍 Checking consistency of image patterns across datasets...')
    image_pattern_keys = set()
    for dataset_name, dataset_config in config['datasets'].items():
        current_keys = set(dataset_config['image_patterns'].keys())
        if not image_pattern_keys:
            image_pattern_keys = current_keys
        elif image_pattern_keys != current_keys:
            logging.error(f'❌ Inconsistent image pattern keys found in dataset {dataset_name}')
            logging.error(f'🔑 Expected keys: {image_pattern_keys}')
            logging.error(f'🔑 Found keys: {current_keys}')
            raise ValueError(f'Inconsistent image pattern keys in dataset {dataset_name}')
    
    logging.info('✅ Image pattern keys are consistent across all datasets.')
    logging.debug(f'🔑 Consistent image pattern keys: {image_pattern_keys}')

    return config, output_dir, data_root, num_folds, list(image_pattern_keys)


def process_datasets(config, output_dir, data_root, num_folds, image_pattern_keys):
    all_missing_data = {}
    dataset_stats = {}
    all_affine_issues = {}

    for dataset_name, dataset_config in config['datasets'].items():
        logging.info(f'🔍 Generating JSON for {dataset_name} dataset...')
        data, missing_data, affine_issues = generate_json_for_dataset(data_root, dataset_name, dataset_config, num_folds, image_pattern_keys)
        json_path = os.path.join(output_dir, f'{dataset_name}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f'💾 Saved JSON file for {dataset_name} at {json_path}')

        all_missing_data[dataset_name] = missing_data
        dataset_stats[dataset_name] = calculate_dataset_stats(data, missing_data)
        all_affine_issues[dataset_name] = affine_issues

    logging.info('🎉 JSON generation process completed successfully.')
    return all_missing_data, dataset_stats, all_affine_issues


def calculate_dataset_stats(data, missing_data):
    return {
        'patients_with_data': len(data['training']),
        'patients_missing_label': len(missing_data['label']),
        'patients_missing_modalities': len(missing_data['modalities'])
    }


def generate_summary(all_missing_data, dataset_stats, all_affine_issues):
    summary = {
        'missing_data': all_missing_data,
        'dataset_stats': dataset_stats,
        'affine_issues': all_affine_issues
    }
    
    # Calculate summary stats across all datasets
    summary_stats = {
        'total_patients_with_data': sum(stats['patients_with_data'] for stats in dataset_stats.values()),
        'total_patients_missing_label': sum(stats['patients_missing_label'] for stats in dataset_stats.values()),
        'total_patients_missing_modalities': sum(stats['patients_missing_modalities'] for stats in dataset_stats.values()),
        'total_datasets': len(dataset_stats)
    }
    summary['summary_stats'] = summary_stats
    
    logging.info('📊 Generated summary stats across all datasets.')
    logging.debug(f'📊 Summary stats: {summary_stats}')
    
    return summary

def save_summary(summary, output_dir):
    logging.info('📝 Preparing to save summary...')
    summary_path = os.path.join(output_dir, 'summary.yaml')
    logging.debug(f'📁 {summary_path = }')
    
    logging.info('🔧 Converting defaultdict to regular dict and processing tuples...')
    for dataset, missing_data in summary['missing_data'].items():
        summary['missing_data'][dataset] = dict(missing_data)
        if 'modalities' in summary['missing_data'][dataset]:
            summary['missing_data'][dataset]['modalities'] = [
                {'patient': patient, 'missing': missing}
                for patient, missing in summary['missing_data'][dataset]['modalities']
            ]
    
    logging.debug(f'🔍 Processed summary: {summary}')
    
    logging.info('💾 Writing summary to file...')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    logging.info(f'✅ Saved summary file at {summary_path}.')
    logging.debug(f'📄 Summary content: {summary}')

    logging.info('🔍 Verifying saved summary...')
    with open(summary_path, 'r') as f:
        loaded_summary = yaml.safe_load(f)
    logging.debug(f'📄 Loaded summary content: {loaded_summary}')
    
    if loaded_summary == summary:
        logging.info('✅ Verification successful: saved and loaded summaries match.')
    else:
        logging.warning('⚠️ Verification failed: saved and loaded summaries do not match.')
        logging.debug(f'🔍 Differences: {set(summary.items()) ^ set(loaded_summary.items())}')


def load_and_print_summary(output_dir):
    summary_path = os.path.join(output_dir, 'summary.yaml')
    with open(summary_path, 'r') as f:
        summary = yaml.safe_load(f)
    logging.info('📊 Summary of data preparation:')
    logging.info(pformat(summary))


def main():
    config, output_dir, data_root, num_folds, image_pattern_keys = load_and_prepare_config()
    all_missing_data, dataset_stats, all_affine_issues = process_datasets(config, output_dir, data_root, num_folds, image_pattern_keys)
    summary = generate_summary(all_missing_data, dataset_stats, all_affine_issues)
    save_summary(summary, output_dir)
    load_and_print_summary(output_dir)


if __name__ == '__main__':
    main()