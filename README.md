# DeepR2cov
A deep representation on heterogeneous drug network, termed DeepR2cov, to discover potential agents for treating the excessive inflammatory response in COVID-19 patients.

# Data description
* Example_metapath: A representative subset of meta paths.
* CMapscore: Connectivity map score based on up- and down-regulated genes of SARS patients for 2439 drug compounds.

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
The network representation model and training regime in DeepR2cov are similar to the original implementation described in "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/)". Therefore, the code of network representation of DeepR2cov can be downloaded from https://github.com/google-research/bert. But BERT uses a combination of two tasks, i.e,. masked language learning and the consecutive sentences classification. Nevertheless, different from natural language modeling, meta paths do not have a consecutive relationship. Therefore, DeepR2cov does not involve the continuous sentences training. If you want to run DeepR2cov, please manually replace the run_pretraining.py in [BERT](https://github.com/google-research/bert) with this file. 
  
* Download the [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model: 12-layer, 768-hidden, 12-heads. 
You can construct a vocab file (vocab.txt) of nodes and modify the config file (bert_config.json) which specifies the hyperparameters of the model.
* Run create_pretraining_data.py to mask metapath sample.  
<pre> python create_pretraining_data.py   \
  --input_file=../example_metapath.txt   \
  --output_file=../tf_examples.tfrecord   \
  --vocab_file=../uncased_L-12_H-768_A-12/vocab.txt   \ 
  --do_lower_case=True   \  
  --max_seq_length=128   \  
  --max_predictions_per_seq=20   \
  --masked_lm_prob=0.15   \ 
  --random_seed=12345   \
  --dupe_factor=5 </pre>
The max_predictions_per_seq is the maximum number of masked meta path predictions per path sample. masked_lm_prob is the probability for masked token.

* Run run_pretraining.py to train a network representation model.
<pre> python run_pretraining.py   \  
  --input_file=../tf_examples.tfrecord   \  
  --output_dir=../RLearing_output   \  
  --do_train=True   \  
  --do_eval=True   \  
  --bert_config_file=../uncased_L-12_H-768_A-12/bert_config.json   \  
  --train_batch_size=32   \  
  --max_seq_length=128   \  
  --max_predictions_per_seq=20   \  
  --num_train_steps=20   \  
  --num_warmup_steps=10   \  
  --learning_rate=2e-5  </pre>
  
* Run extract_features.py extract_features.py to attain the low-dimensional representation vectors of vertices.
<pre> python extract_features.py   \  
  --input_file=../node.txt   \  
  --output_file=../output.jsonl   \  
  --vocab_file=../uncased_L-12_H-768_A-12/vocab.txt   \  
  --bert_config_file=../uncased_L-12_H-768_A-12/bert_config.json   \  
  --init_checkpoint=../RLearing_output/bert_model.ckpt   \  
  --layers=-1,-2,-3,-4   \  
  --max_seq_length=128   \  
  --batch_size=8 </pre>

* Run PDI_drug_cov.py to predict of the confidence scores between drugs and TNF-α/IL-6.  
<pre> python PDI_drug_cov.py </pre>

* Run top_rank.py to select top 20 high-confidence drugs binding to TNF-α and IL-6, respectively.   
<pre> python top_rank.py   </pre>

# Contacts
If you have any questions or comments, please feel free to email: xqw@hnu.edu.cn.
