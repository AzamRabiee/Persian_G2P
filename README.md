# Persian_g2p: A seq-to-seq model for Persian (Farsi) Grapheme To Phoneme mapping

* forked from [g2pE](https://github.com/Kyubyong/g2p), training is implemented using pytorch; but numpy is used in the inference.  

Persian_g2p converts Persian (Farsi) graphemes (spelling) to phonemes (pronunciation), which can be used as the 
pre-processing step for text-to-speech (TTS). Persian language, alike English, is in the category of complex
languages, as Persian words are not always written in the way that they are pronounced. 

Thanks to the useful repositories that are helpful in surviving Persian language and making this repo: 
- [Tihu dictionary](https://github.com/tihu-nlp/tihudict) is a Persian pronunciation dictionary, useful for g2p mapping. 
However, it does not contain all the words. 
Thus, in this repo a sequence-to-sequence network with cross-entropy loss is trained on the tihu dictionary to handle OOVs. 
-  [hazm](https://github.com/sobhe/hazm) is used for the Persian text pre-processing and normalization. It was helpful 
in Word tokenizing and text cleaning, e.g. replacing ي and ك with ی and ک , respectively.
-  [num2fawords](https://github.com/5j9/num2fawords) package converts numbers to Persian words, 
e.g. 12.4 --> دوازده و چهار دهم  

## usage
### installing dependencies
 1. install python 3.
 2. install requirements, as follows:
   ```
   pip install -r requirements.txt
   ```
### using a pre-trained model
Run the following command with your desired `<Persian-text>` to see its pronunciation. 
   ```
   python g2p.py --text <Persian-text>
   ```
For instance, this is the output for replacing the `<Persian-text>` with «زان یار دلنوازم شکریست با شکایت»: 
   ```
   python g2p.py --text زان یار دلنوازم شکریست با شکایت
   >>> zAn yAr delnavAzam Sokrist bA SekAyat 
   ```

Please note that the pronunciation of the words that are already available comes from
 the dictionary; while the pronunciation of the new words (OOV) are predicted using the trained model
 i.e. «دلنوازم» in this example.
 
 You can also specify your checkpoint with `--checkpoint` option, as follows:   
   ```
   python g2p.py --text <Persian-text> --checkpoint <path-to-the-checkpoint>
   ```

### training model using alternative word-pronunciation dictionary
The network architecture is actually a seq2seq GRU-RNN, in which the last hidden vector 
of the encoder is fed to the decoder, as the below figure.  

![network architecture, seq2seq](https://github.com/AzamRabiee/Persian_G2P/blob/master/imgs/network.png)

You may take a look at the [hparams.py](https://github.com/AzamRabiee/Persian_G2P/blob/master/hparams.py) for default network parameters. 
The following command starts the training process, making a log folder, naming 
`logs-<run-name>`, for storing checkpoint, results, as well as logs of loss and PERs. 
   ```
    python train.py --name <run-name> --dictionary <path-to-the-word-pronunciation-dictionary>
   ```
## TODO:
- improving the performance; currently the PER is 3.9% on test set but not enough.
- adding informal words (slang, such as چاکرتیم- واس ماس and broken language, e.g. نمیشه- میرم- بگم-همشون) to the dictionary

## References

If you use this code for research, please cite:

```
@misc{Persian_g2p2019,
  author = {Rabiee, Azam},
  title = {Persian g2p},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AzamRabiee/Persian_G2P}}
}
```
