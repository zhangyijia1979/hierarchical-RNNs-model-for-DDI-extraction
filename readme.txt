Hierarchical RNNs model-based DDI extraction
================================================================================

Hierarchical RNNs model-based DDI extraction aims to detect and classify the drug-drug interaction in biomedical texts, which has been the state-of-the-art methods for DDI extraction task. 

In this project, we will provide our implementations of Hierarchical RNNs model and DDI extraction 2013 corpus. It is developed with Keras 1.0.2 and python 2.7.

This package contains an implementation of DDI extraction tool based on hierarchical RNNs model described in the paper "Drug-drug interaction extraction via hierarchical RNNs on sequence and shortest dependency paths". In addition, this package contains DDI corpora, which is from the DDIs Extraction Challenge 2013 task.

This is research software, provided as is without express or implied warranties etc. see licence.txt for more details. We have tried to make it reasonably usable and provided help options, but adapting the system to new environments or transforming a corpus to the format used by the system may require significant effort. 

The details of related files are described as follows:

DDIextraction2013: the folder that contains the DDIextraction 2013 corpora including train data and test data. If you are intrested in the DDI extaction task, you can find more information about the DDIextraction 2013 and 2011 from http://labda.inf.uc3m.es/doku.php?id=en:labda_ddicorpus.

PreprocessData: the folder that contains three pkl files including train.pkl.gz, test.pkl.gz and vec.pkl.gz,respectively.

Sourcecode: the folder that contains the source code and a saved model.


============================ QUICKSTART ========================================

The main requirement of the software is Keras 1.0.2, Theano 0.9.0, python 2.7 and numpy.

User can use hierarchi_Rnns.py to automatic extract DDIs from pkl files which are preprocessed from xml format corpora.

Since the model contains multiple layer, it generally need some time to train. If the users have no time or GPU to train model, the saved model in the Sourcecode can be loaded to test.

Note that user maybe change the path in hierarchi_Rnns.py and three pkl files.



