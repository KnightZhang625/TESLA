# Named Entity Recognition     
This module is a embedder based on [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma and Hovy (2016)](https://arxiv.org/abs/1603.01354). The model uses CNNs to embed character representation of words in a sentence and stacked bi-direction LSTM layers to embed the context of words and characters.

## Bash Scripts   
```shell
./train.sh train the model. 
./package.sh package the model to the pb file.  
./infer.sh  inference according to the pb file.
```

## Run Example on the CONLL2000 dataset  
- 1. Convert the txt format data to binary format   
      The CONLL2000 dataset is in the datasets directory; run [data_pipeline.py](https://github.com/KnightZhang625/TESLA/blob/master/tesla/utils/data_pipeline.py) to covert the data, the code is at the below.  
- 2. Create the dictionary, run [data_pipeline.py](https://github.com/KnightZhang625/TESLA/blob/master/tesla/nlp_task/ner/data_pipeline.py)(different from the above one). Please uncomment the below code.  
- 3. Change the config.py under the ner directory.  
- 4. Run ./train.sh to train the model.  
- 5. Run ./package.sh to package the model to the pb format.  
- 6. Run ./infer.sh to make the inference on the toy example at the bottom of the code.  