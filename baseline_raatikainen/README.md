# Dyslexia detection public source codes

Research scripts originally used in the Master's Thesis "Automatic
detection of developmental dyslexia from eye movement data" 
( http://urn.fi/URN:NBN:fi:jyu-201906103105 ). Re-publication of main
findings in a scientific journal is being considered.

We publish these scripts to improve the transparency of our research,
even though we acknowledge that the results cannot be reproduced
without the original data that we cannot make public due to privacy
decisions made before making the experiments with human participants.

Files:

```
createMatrixAvgFeatures.py                        - Creation of derived features
trainParameterSearch.py                           - Algorithm and grid search for SVM
trainParameterSearchRandomForestHistFeatImport.py - Algorithm and grid search for RF
```

The codes are pending future workforce recruitments for further
refactoring and taking to a next level. As of now, the RF code is
copy-paste-modify from the SVM one. Lines have been matched using
empty lines so that the differences between the two can be readily
observed side-by-side or using for example the following unix command:

```
diff trainParameterSearch.py trainParameterSearchRandomForestHistFeatImport.py
```

The first results published in the cited thesis were produced by
repeated runs of modified versions of these scripts. Critical parts of
development history are documented in comments.

Author:

Peter Raatikainen and his thesis supervisors, 2018

Correspondence:

Paavo Nieminen <paavo.j.nieminen@jyu.fi> (one supervisor of said thesis)
