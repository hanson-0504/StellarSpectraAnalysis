import os
import sys
import tempfile
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import setup_env, read_text_file, load_config

def test_setup_env_creates_directories():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.yaml")
        config_content = """\
directories:
  data: "data/"
  spectral: "data/spectral_dir/"
  labels: "data/label_dir/"
  models: "data/models/"
  tuners: "data/kt_tuner_dir/"
  output: "output/"
  logs: "logs/"
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert "directories" in config

        cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            setup_env(config_path)
            # Assert all expected directories were created
            assert os.path.isdir("data")
            assert os.path.isdir("data/spectral_dir")
            assert os.path.isdir("data/label_dir")
            assert os.path.isdir("data/models")
            assert os.path.isdir("data/kt_tuner_dir")
            assert os.path.isdir("output")
            assert os.path.isdir("logs")
        finally:
            os.chdir(cwd)
            
            
def test_read_text_file():
    test_lines = ["Teff", "logg", "#[Fe/H]"] # [Fe/H] should not be read
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file_path = os.path.join(temp_dir, "params.txt")
        with open(test_file_path, 'w') as f:
            f.write("\n".join(test_lines))
        result = read_text_file(test_file_path)
        assert result == ["Teff", "logg"]
        