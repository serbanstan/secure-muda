Code to run experiments from Secure Domain Adaptation with Multiple Sources.

Our code builds on the implementations in https://sites.google.com/view/simpal 
(Your Classifier can Secretly Suffice Multi-Source Domain Adaptation)
,https://github.com/jindongwang/transferlearning (Transfer Learning),
https://github.com/eifuentes/swae-pytorch (Sliced Wasserstein Autoencoder) and
https://github.com/driptaRC/DECISION (Unsupervised Multi-source Domain Adaptation Without Access to Source Data)
.

Pre-trained features should be added under data/pretrained-features. These features can be generated automatically via 
the code in code/feature-gen-code once the data/ folder has been populated. Due to space constraints for this submission,
we only provide pre-trained features for one of the datasets.

To run experiments, simply go to code/, and call run_pipeline.py . An example run can take the form:

python run_pipeline.py --dataset image-clef --num_exp 3 --gpu_id 0

More documentation on how to use this script can be found inside run_pipeline.

* main.py calls the E2E training routine, and then calls evaluate_model which computes target results
* config.py and config_populate.py correspond to populating necessary information regarding dataset (#classes, 
	network architecture, im. resolution etc.)
* exp_select.py is an addition to config, and is responsible for selecting the experiment to be run.
* evaluate_model.py generates target results for different choices of w
* trainer.py implements the MultiSourceTrainer class, which first trains a model on source data, then performs adaptation.
           This file represents the bulk of the algorithm described in the manuscript.
* guassian_utils.py implements logic related to learning and sampling from the intermediate GMM distribution
* runner.py allows running a single task for a dataset. For example
python runner.py --dataset image-clef --task PC_I --gpu_id 0 --num_exp 5
* dataset.py implements reading in a dataset into pytorch
* metrics.py implements wasserstein distance alongside accuracy metrics
* net.py implements the network logic
* customlayers.py implements the individual network modules (e.g. fully connected module, classifier module etc.)

Following a succesful run, dataset statistics can be found in the summaries/ folder, and model weights can be 
found in the weights/ folder. The current code allows the model to train several tasks in parallel accross multiple 
gpus. 

