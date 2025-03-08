# Brain-Tumor-MRI-Classification 

## **Project Overview**  
This project aims to build a **Convolutional Neural Network (CNN)** to detect tumors in MRI scans. If there’s not enough data available, I’ll generate synthetic MRI images using **Generative Adversarial Networks (GANs)** to improve model performance. 

## Project Guidelines

### **1. Understanding the Problem & Collecting Data**  
For this project, I’ll need a dataset of MRI images labeled as "glioma," "meningioma," "pituitary", or "notumor". I’ll start by researching the problem and collecting data from Kaggle.

Having already collected the data, it is stored in the `dataset` folder. The dataset is organized as follows:
```
dataset
│
└───glioma
│   │   Tr-gb_0010.jpg
│   │   Tr-gl_0011.jpg
│   │   ...
│
└───meningioma
│   │   Tr-me_0010.jpg
│   │   Tr-me_0011.jpg
│   │   ...
│
└───pituitary
│   │   Tr-pi_0010.jpg
│   │   Tr-pi_0011.jpg
│   │   ...
│
└───notumor
    │   Tr-no_0010.jpg
    │   Tr-no_0011.jpg
    │   ...
```

### **2. Data Preprocessing**  
Before training, I’ll clean and prepare the MRI images:  
- Resize them to a consistent shape.  
- Normalize pixel values (e.g., scale between 0 and 1).  
- Apply contrast enhancement if needed.  
- Split the dataset into **training, validation, and test sets**.

### **3. Building the CNN Model**  
I’ll design a **CNN architecture** that can classify MRI images. 
Several architectures can be used, such as **LeNet, AlexNet, VGG, ResNet, or Inception**.
In this project, I’ll start with a simple architecture and then perhaps experiment with more complex models.
Key decisions include:  
- Choosing convolutional layers for feature extraction.  
- Using dropout and pooling layers to improve generalization.  
- Selecting an optimizer (like Adam) and a loss function (categorical cross-entropy).  
- Compiling the model and training it on the training set.

### **4. Evaluating Performance**  
To check how well the model works, I’ll:  
- Measure accuracy, precision, recall, and F1-score.  
- Plot a confusion matrix to see where the model makes mistakes.  
- Tune hyperparameters if needed to improve results.  

### **5. Data Augmentation (If Needed)**  
If the dataset is too small, I’ll start with traditional augmentations, like rotating, flipping, or scaling images. If that’s not enough, I’ll move to **synthetic data generation using GANs**.  

### **6. Generating Synthetic MRI Scans with GANs**  
I’ll train a **GAN (e.g., DCGAN or CycleGAN)** on real MRI scans that we have in the dataset folder. The GAN will learn the distribution of the data and be able to generate realistic synthetic images, after training. These new images will be added to the dataset to improve the CNN’s accuracy.  

### **7. Final Training & Deployment**  
Once I’ve augmented the dataset, I’ll retrain the CNN and evaluate it again.

---  

### Essential models for image classification

Most common CNN models among which we can choose the adapted model for our project:

| Model                | Year  | Parameters (Millions) | Pretrained Available? | Recommended Dataset Size | Input Image Format |
|----------------------|------|---------------------|----------------------|-------------------------|--------------------|
| **LeNet-5**         | 1998 | 0.06                | No                   | ~10K+ (e.g., MNIST)    | 32×32 (Grayscale)  |
| **AlexNet**         | 2012 | 60                  | Yes (ImageNet)       | ~1M+ (e.g., ImageNet)  | 227×227 (RGB)      |
| **VGG-16**          | 2014 | 138                 | Yes (ImageNet)       | ~1M+ (e.g., ImageNet)  | 224×224 (RGB)      |
| **VGG-19**          | 2014 | 144                 | Yes (ImageNet)       | ~1M+ (e.g., ImageNet)  | 224×224 (RGB)      |
| **GoogLeNet (Inception v1)** | 2014 | 5 | Yes (ImageNet) | ~500K+ | 224×224 (RGB) |
| **Inception v3**    | 2015 | 23.8                | Yes (ImageNet)       | ~1M+                     | 299×299 (RGB)      |
| **ResNet-18**       | 2015 | 11.7                | Yes (ImageNet)       | ~500K+                  | 224×224 (RGB)      |
| **ResNet-50**       | 2015 | 25.6                | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |
| **ResNet-101**      | 2015 | 44.5                | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |
| **ResNet-152**      | 2015 | 60.2                | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |
| **DenseNet-121**    | 2017 | 8.0                 | Yes (ImageNet)       | ~500K+                   | 224×224 (RGB)      |
| **DenseNet-169**    | 2017 | 14.3                | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |
| **DenseNet-201**    | 2017 | 20.0                | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |
| **MobileNetV1**     | 2017 | 4.2                 | Yes (ImageNet)       | ~100K+                   | 224×224 (RGB)      |
| **MobileNetV2**     | 2018 | 3.4                 | Yes (ImageNet)       | ~100K+                   | 224×224 (RGB)      |
| **EfficientNet-B0** | 2019 | 5.3                 | Yes (ImageNet)       | ~100K+                   | 224×224 (RGB)      |
| **EfficientNet-B7** | 2019 | 66                  | Yes (ImageNet)       | ~1M+                     | 600×600 (RGB)      |
| **ConvNeXt-Tiny**   | 2022 | 28                  | Yes (ImageNet)       | ~1M+                     | 224×224 (RGB)      |

In our case, we can start with a simple model we build ourselves. Then, we will choose a more complex model to continue with. But since I do not have a GPU, the models can't be too complex to train. This restricts us to models that do not have too many parameters: LeNet-5, MobileNetV1, MobileNetV2, EfficientNet-B0.

Apart from LeNet-5, the models all expect an input image size of 224×224x3. We will therefore need to convert our images to this format before training the models.

