### Project Name: **MNIST Autoencoder and k-Nearest Neighbors (kNN) Analysis**

#### Project Description:
This project explores machine learning techniques on the MNIST dataset of handwritten digits. It focuses on two key algorithms: an **Autoencoder** for unsupervised feature learning and a **k-Nearest Neighbors (kNN)** classifier for supervised learning. The goal is to understand the strengths and limitations of both methods in digit recognition tasks.

#### Objectives:
1. **Understand Autoencoders**: Implement and analyze autoencoders to learn compressed representations of the MNIST data.
2. **Implement kNN Classifier**: Develop a kNN classifier to predict digit labels based on the Euclidean distance between feature vectors.
3. **Compare Performance**: Evaluate and compare the performance of the autoencoder for feature extraction and the kNN classifier for classification.

#### Technical Implementation:

1. **Data Preparation**:
   - Load the MNIST dataset using provided functions (`loadMNISTImages.m` and `loadMNISTLabels.m`).
   - Normalize pixel values and reshape the dataset for processing.

   ```matlab
   [images, labels] = loadMNIST('path_to_mnist');
   images = double(images) / 255; % Normalize
   ```

2. **Autoencoder Implementation**:
   - Construct a multi-layer perceptron as an autoencoder, ensuring the input matches the target.
   - Train the autoencoder using MATLAB’s `trainAutoencoder` function.

   ```matlab
   nh = 100; % Number of hidden units
   myAutoencoder = trainAutoencoder(images, nh);
   encodedData = encode(myAutoencoder, images);
   ```

3. **k-Nearest Neighbors Classifier**:
   - Use the encoded representations from the autoencoder as input features for the kNN classifier.
   - Implement the kNN algorithm using the `kNNClassifier.m` function, setting a suitable value for k (e.g., k=3).

   ```matlab
   predictedLabels = kNNClassifier(encodedData, labels, k);
   ```

4. **Model Evaluation**:
   - Split the MNIST dataset into training and test sets.
   - Evaluate the autoencoder’s ability to reconstruct the images and the kNN classifier's accuracy in predicting digit labels.
   - Create confusion matrices and accuracy metrics to assess performance.

5. **Visualization**:
   - Visualize the original versus reconstructed images from the autoencoder to gauge performance.
   - Plot confusion matrices for the kNN classifier to analyze classification results.

   ```matlab
   % Visualize original and reconstructed images
   figure; 
   subplot(1,2,1); imshow(reshape(originalImage, 28, 28)); title('Original');
   subplot(1,2,2); imshow(reshape(reconstructedImage, 28, 28)); title('Reconstructed');
   ```

6. **Documentation and Reporting**:
   - Document findings in a comprehensive report, detailing methodologies, experimental setups, results, and insights from both the autoencoder and kNN analyses.
   - Include code snippets and visualizations to illustrate the outcomes effectively.

#### Key Files:
- **kNN.m**: Implementation of the kNN algorithm.
- **kNNClassifier.m**: Main function to classify images using kNN.
- **loadMNIST.m**: Loads the MNIST dataset.
- **loadMNISTImages.m**: Loads the images from the dataset.
- **loadMNISTLabels.m**: Loads the corresponding labels.

This project offers a hands-on approach to understanding neural networks and classification algorithms, providing valuable insights into feature extraction and supervised learning techniques.
