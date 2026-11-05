# The first project cell (cell 2) of notebooks/Flip-Classifier-Training-Notebook.ipynb at the v1.5.0 tag, unchanged.

from moseq2_app.flip.widget import FlipClassifierWidget
from moseq2_app.flip.train import create_training_dataset, train_classifier, save_classifier, CleanParameters
from moseq2_app.util import read_yaml
import panel as pn
pn.extension()

# specify paths
config_path = '/path/to/config.yaml'
input_dir = '/path/to/aggregate_results' # Specify the data folder
model_path = '/path/to/flip/new-flip-tst.pkl' ## e.g. ./flip-classifier-azure-ephys.pkl

FF = FlipClassifierWidget(input_dir)
