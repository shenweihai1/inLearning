## Fine-tune models
### fine\_tune.ipynb
implementations of transition from one dataset to another formatted dataset for fine-tuning
#### procedure
1. convert the `train.json` into the format described below. In this case, we put back the correct answer into the question, then split this question with correct answer into two sentence, first sentence as the first line, second sentence put in the second line. please refer to the [lm_finetuning](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples/lm_finetuning)
2. all the code based on library: https://github.com/huggingface/pytorch-pretrained-BERT

### fine-tune-bash.ipynb
python command to fine-tune the model
1. Simple fine-tune 
```
python examples/lm_finetuning/simple_lm_finetuning.py 
    --train_batch_size 2 
    --train_corpus 
    ../corpus.txt 
    --bert_model 
    bert-base-uncased 
    --output_dir finetuned_lm 
    --do_train 
```
2. fine-tuned model can be downloaded from https://drive.google.com/open?id=1zcJ_6pm-cUMdoJ3ZwGxVx4iv_zSukPkU

### base_line.ipynb
using bert to predict the next sentence with pre-trained model `bert-base-uncased` to get a base_line accuracy
#### Procedure
1. convert the `dev.json` to four sentences: (question, correct answer), (question, wrong answer). 
For instance, original sentence is **"Tommy glided across the marble floor with ease, but slipped and fell on the wet floor because `_____` has more resistance. (A) marble floor (B) wet floor"** where correct answer is wet floor. We can convert this sentenct into: (**"Tommy glided across the marble floor with ease, but slipped and fell on the wet floor because `_____` has more resistance."**, **"wet floor"**), (**"Tommy glided across the marble floor with ease, but slipped and fell on the wet floor because `_____` has more resistance."**, **"marble floor"**)
2. using bert to predict the next sentence. In this case, we use correct and wrong answer as the next sentence to get confidence separately, pick up the higer one as the predicted answer.
3. accuracy on dev is `55.4 %`, on test is `53.3 %` using pre-trained model `bert-base-uncased`

### feature_extract.py
this script extract the corresponing features from pre-processed text corpus using fine tuned bert model. The filename as in format `'output/corpus_*_*.txt'`. For example, 'corpus_train_true.txt' contains all the statements with true answers of train set. specify the dataset by change the value of 'par' variable.  the finename of pretrained model is called `'pretrained_hacked.model'`

### classifier.py
This script uses tensorflow to build a Neural Network model to classify the output sample from feature.py. Generally speaking, this a text binary classification task.
1. Prepare the data
As the input to the neural network model should be in the same lengthy, in function getMatrix we padded the shorter sentence with 0 to standardize the lengths.
2. Neural Network model
We used a neural network model with a sequential structure to do the classification. With the neural network model, for each input sentence, we can get a probability on the confidence if this is a statement with correct answer. The structure of the neural network model is `Dense - Batch Normalization - tanh - Dense - Batch Normalization - tanh - softmax - binary cross entropy loss`. We used `Adam` to optimizations the model on batches.
