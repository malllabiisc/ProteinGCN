## ProteinGCN: Protein model quality assessment using Graph Convolutional Networks
 

Source code for the paper: **ProteinGCN: Protein model quality assessment using Graph Convolutional Networks**

![](./overview.png)

**Overview of ProteinGCN:** *Given a protein structure, it first generates a protein graph and uses GCN to learn the atom embeddings. Then, it pools the atom embeddings to generate residue-level embeddings. The residue embeddings are passed through a non-linear fully connected layer to predict the local scores. Further, the residue embeddings are pooled to generate a global protein embedding. Similar to residue embeddings, this is used to predict the global score.*

### Dependencies

- Compatible with PyTorch 1.0 and Python 3.x.
- Dependencies can be installed using the `requirements.txt` file.

### Dataset:

- We use Rosetta-300k to train the ProteinGCN model and test it on both Rosetta-300k and CASP13 dataset for local(residue) and global Quality Assessment predictions.

### Training model:

- Install all the requirements by executing `pip install -r requirements.txt.`

- Install required protein `.pdb` processing library by executing `sh preprocess.sh` which clones and installs [this](https://github.com/gjoni/mylddt) github repository.

- Next execute `python preprocess_pdb_to_pkl.py` script which creates the required `.pkl` files from the dataset to be used for model training. It defaults to a sample dataset provided with the code at `./data/`. To use the original datasets, please change the paths accordingly.

- To start a training run:
  ```shell
  python train.py trial_run --epochs 10
  ```
  Once successfully run, this creates a folder by the name `trial_run` under the path `./data/pkl/results/` which contains the test results `test_results.csv` and best model checkpoint `model_best.pth.tar`. Rest of the training arguments and the defaults can be found in `arguments.py`.

- To get the final pearson correlation scores, run:
  ```shell
  python correlation.py -file ./data/pkl/results/trial_run/test_results.csv
  ``` 

- We have published our best ProteinGCN model that was trained on Rosetta-300k dataset. To run the pretrained model on the sample data, execute:
  ```shell
  python train.py trial_testrun --pretrained ./pretrained/pretrained.pth.tar --epochs 0 --train 0 --val 0 --test 1
  ```

For any clarification, comments, or suggestions please create an issue or contact [Soumya](https://soumyasanyal.github.io/).
