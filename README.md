# DeepR2cov

A deep representation on heterogeneous drug network, termed DeepR2cov, to discover potential agents for treating the excessive inflammatory response in COVID-19 patients.

# Data description
* CMapscore: Connectivity map score based on up- and down-regulated genes of SARS patients for 2439 drug compounds.
* Example_metapath: A representative subset of meta paths.


# Requirements
DeepR2cov is tested to work under:
* Python 3.6  
* Tensorflow 1.1.4
* tflearn
* numpy 1.14.0
* sklearn 0.19.0

# Quick start
* Download the source code of [BERT](https://github.com/google-research/bert). 
* Manually replace the run_pretraining.py
The network representation model and training regime in DeepR2cov are similar to the original implementation described in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". Therefore, the code of network representation of DeepR2cov can be downloaded from https://github.com/google-research/bert. But BERT uses a combination of two tasks, i.e,. masked language learning and the consecutive sentences classification. Nevertheless, different from natural language modeling, meta paths do not have a consecutive relationship. Therefore, DeepR2cov does not involve the continuous sentences training. If you want to run DeepR2cov, please manually replace the run_pretraining.py in [BERT](https://github.com/google-research/bert) with this file. 
  
* Download the models [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads. 
You can construct a vocab file (vocab.txt) of nodes and modify the config file (bert_config.json) which specifies the hyperparameters of the model.
* Run create_pretraining_data.py to mask metapath sample.  
python create_pretraining_data.py   \  
--input_file=../example_metapath.txt   \  
--output_file=../tf_examples.tfrecord   \  
--vocab_file=../uncased_L-12_H-768_A-12/vocab.txt \  
--do_lower_case=True   \  
--max_seq_length=128   \  
--max_predictions_per_seq=20   \  
--masked_lm_prob=0.15   \  
--random_seed=12345   \  
--dupe_factor=5   \  
The max_predictions_per_seq is the maximum number of masked LM predictions per sequence. masked_lm_prob is the probability for masked token. You should set this to around max_seq_length*masked_lm_prob.

* Run run_pretraining.py to attain a network representation model. Options are:\
python run_pretraining.py \
--input_file=../tf_examples.tfrecord \
--output_dir=../RLearing_output \
--do_train=True \
--do_eval=True \
--bert_config_file=../uncased_L-12_H-768_A-12/bert_config.json \
--train_batch_size=32 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=20 \
--num_warmup_steps=10 \
--learning_rate=2e-5 

* Run extract_features.py extract_features.py to attain the low-dimensional representation vectors of vertices. Options are:\
python extract_features.py \
--input_file=../node.txt \
--output_file=../output.jsonl \
--vocab_file=../uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=../uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=../RLearing_output/bert_model.ckpt \
--layers=-1,-2,-3,-4 \
--max_seq_length=128 \
--batch_size=8 

* Run PDI_drug_cov.py PDI_drug_cov.py to predict of drug-TNF-α/IL-6 confidence scores. Options are: \
  python PDI_drug_cov.py	-n 1 -k 512 \
  n is global norm to be clipped, and k is the dimension of project matrices. 

* Run top_rank.py to select top 20 high-confidence drugs binding to TNF-α and IL-6, respectively. \
  python top_rank.py 

# Contacts
If you have any questions or comments, please feel free to email:xqw@hnu.edu.cn.
