import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.datasets import fetch_olivetti_faces

class Face_Recognition():
	##Declane classes
	def __init__(self):
		return 
	def plot_samples(self):
		#Load some image samples
		plt.figure(figsize=(20,25))
		for i in range(100,120):
			plt.subplot(4,5,i-99)
			plt.imshow(images_dataset[i], cmap="gray")
		plt.show()


	# PCA transformation and reduce the dimension from 4096 to 50
	##Define the number of components 'n'=100
	def PCA_analysis(self,X_train,X_test):
		n=144
		pca = PCA(n_components=n, whiten=True)
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		plt.plot(range(1,(n+1)), pca.explained_variance_ratio_.cumsum())
		plt.title('Explained Variance',fontsize=15)
		plt.xlabel('Number of Principle Components', fontsize=10)
		plt.savefig("../RESULTS/PCA_components.png")
		plt.show()
		
		return X_train, X_test, pca

	def predict_single_image(self,model, image_pca):
		# Fazer a previsão diretamente, já que a imagem já está no formato PCA
		predicted_label = model.predict(image_pca)
		return predicted_label[0]


obj_face_recognition=Face_Recognition()

#Split data on: input, target and images
data=fetch_olivetti_faces()
print(data.keys())

images_dataset=data.images
target= data.target
input_data=data.data

#plot_samples()

#Extract features
#import PCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(input_data, target, shuffle=True)

#Make the PCA analysis
X_train, X_test, pca = obj_face_recognition.PCA_analysis(X_train,X_test)
print(X_test.shape)
#PCA_analysis(X_train,X_test)

#Train and evaluate model
from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train)

score=model.score(X_test, y_test)
print("Score: {}".format(score))

##Test id
detected_face_ids = model.predict(X_test)


# Escolha uma imagem aleatória do conjunto de teste (já transformado pelo PCA)
random_index = np.random.randint(len(X_test))
single_image_pca = X_test[random_index].reshape(1, -1)

predicted_label = obj_face_recognition.predict_single_image(model, single_image_pca)

# Visualizar a imagem original e a previsão
plt.imshow(X_test[random_index].reshape(12,12), cmap="gray")
plt.title(f"Predicted: {predicted_label}, True: {y_test[random_index]}")
plt.show()

