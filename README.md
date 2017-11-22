# Intelligent Feature Generation and Image Classification 

The various methods I used to generate features 
* Tradition image processing libraries ( openCV / PIL )
* Neural Networks ( CNN using theano/ keras )

## 1. Feature generation using traditional computer vision packages
( not covered in this wiki , please refer to the other repository - "Face detection" )

In additional to the feature generation using default Image processing libraries ( openCV, PIL etc in python), CNNs can be trained to identify objects and can be generalized to just detect features like 
* shape of the object detected
* color
* orientation 
* relative size
* if the object is in the foreground or background 

## 2. Image Classification using Neural Networks 
### _( Case : Online Apparel portal dress classification )_ (Using tensor flow/theano and kearas , python)

**Step 1 :** Collect training samples ( Here jeans v/s Indian traditional dress )

I used as low as 150 images for both classes to train.
![Indian Dresses Training Set](https://github.com/tomtillo/ImageClassification/blob/master/indian2.JPG)
_Sample training subset of Indian dresses_


![Jeans Training set](https://github.com/tomtillo/ImageClassification/blob/master/jeans2.JPG)
_Sample training subset of jeans_


**Step 2 :** Build the train model with epoch = 1 


***

```
Epoch 1/1
300/300 [==============================] - 9s - loss: 0.3757 - acc: 0.8967 - val_loss: 1.1921e-07 - val_acc: 1.0000

Running epoch: 1

Epoch 1/1
44/300 [===>..........................] - ETA: 4s - loss: 1.1921e-07 - acc: 1.0
108/300 [=========>....................] - ETA: 2s - loss: 1.1921e-07 - acc: 1.0
172/300 [================>.............] - ETA: 1s - loss: 1.1921e-07 - acc: 1.0
236/300 [======================>.......] - ETA: 0s - loss: 1.1921e-07 - acc: 1.0
300/300 [==============================] - 5s - loss: 1.1921e-07 - acc: 1.0000 -
val_loss: 1.1921e-07 - val_acc: 1.0000 

```

***


**Step 3 :** Run a sample validation  
Validation size = 15 images ( for both classes ) 
Got a validation accuracy of 100% - ( after running the training several times )


Step 4 : Train with epoch = 100 
Build a train model running it on 100 epochs. 
and save the model . 

Step 5 : Testing the model .
Tested it on a new random sample set ( 20 images ) from the portal .

Result : 
The model gave an alarming 100% accuracy ( Leads me to obvious problems of over-fitting and bad sampling)
![](https://github.com/tomtillo/ImageClassification/blob/master/result_cloths.JPG)

**Step 6: Model tuning & Next steps**
My dis-satisfaction with the the 100% accuracy is that of _**over-fitting**_ to the training set.
Task 1 : Identify images that will be wrongly classified 
Task 2 : Retrain with divergent training sample set 
Task 3 : Retrain on multiple classes
Task 4: Rather than absolute probabilities ( categorical 1/0 ) , assign relative probabilities (softmax)



