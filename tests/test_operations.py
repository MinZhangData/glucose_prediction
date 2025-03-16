import pytest
import os
from src.data_processing import process_patient_data

@pytest.fixture
def actual_json_file():
    """Return the path of the actual JSON file."""
    return "data/raw/OhioT1DM/2018/train/json/559-ws-training.json"

def test_process_patient_data_with_real_json(actual_json_file):
    """Test process_patient_data by reading an actual JSON file."""
    data_dir = os.path.dirname(actual_json_file)
    patient_file = os.path.basename(actual_json_file)
    output_dir = "data/processed"  # You can modify this to a different test output directory

    # Run the data processing function
    process_patient_data(data_dir, patient_file, output_dir)

    # ✅ Add assertions here to check if the output meets expectations
    assert os.path.exists(output_dir), "Output directory was not created"
