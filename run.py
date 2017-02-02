

#!/bin/bin/python3.4
'''===================== Logiciel de catégorisation d'articles ======================

-> Créé dans le cadre du TIPE 2014/2015 par David Laloum et Arthur Pesah
-> Prend en paramètre des articles en format texte situés dans le dossier "articles_train"
-> Deux modes :
	- Prédiction sur un lien
	- Prédiction statistique sur un nombre donné d'articles du même dossier, non pris en compte dans l'entraînements
-> Pour tester ce programme, 3500 articles extraits de Wikipedia ont été téléchargés. 
-> Ils font parties de 4 catégories : science, sport, art et politique.
-> Testé avec Python 3.4.0, scikit_learn-0.16.1, PyStemmer-1.3.0, BeautifulSoup 4.2.1, 

======================================================================================
'''
from glob import glob
import os
import time

from Stemmer import *
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # Model
from sklearn import svm
from sklearn import metrics # Scores

from tkinter import *
from tkinter import filedialog

import urllib.request as urllib
import urllib.parse as parse
from bs4 import BeautifulSoup

if (os.name == "nt"):
       filesep = '\\'
else:
       filesep = '/'
       
CATEGORY = ["science","sport","art","politique"]

def index_of_category(category):
	''' Transforme le nom d'une categorie en un id (exemple "science" -> 1)'''
	print (category)
	index = 0
	try:
		while CATEGORY[index] != category:
			index += 1
	except:
		print ("[-] Error : category not found")
		exit(0)
	return index
		
def url_to_text (url):
	''' Utile pour la lecture d'un lien http : transforme le lien en un fichier texte lisible par la machine'''
	
	html = urllib.urlopen(url).read()
	soup = BeautifulSoup(html)

	for script in soup(["script", "style"]):
		script.extract()

	text = soup.get_text()
	lines = (line.strip() for line in text.splitlines())
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	text = '\n'.join(chunk for chunk in chunks if chunk)
	return text 






st = Stemmer("fr")
class StemmedVectorizer(TfidfVectorizer):
	''' Classe nécessaire pour prendre en compte le tf-idf des mots, et non juste leur fréquence'''
	def build_analyzer(self):
		analyzer=super(StemmedVectorizer,self).build_analyzer()
		return lambda doc:(st.stemWord(word) for word in analyzer(doc))

class Interface (Frame):
	''' Classe principal du logiciel qui contient toutes les fonctions associées à des boutons
	-> 5 méthodes, décrites dans leur doc_string respectifs :
		− __init__()
		− openfilename()
		− training()
		− predicting()
		− testing()'''
		
	def __init__ (self, window):
		''' Met en place l'interface (fenêtres, frames, textes, boutons...)'''
		
		Frame.__init__ (self, window)
		window.wm_title("Categorisation d'articles")
		self.pack(fill=BOTH)
		
		# Création d'une frame pour le training en haut, et pour la prediction en bas
		
		self.frame_training = LabelFrame (self,text="Entrainement")
		self.frame_training.grid(row=0,pady=10,padx=10,sticky="W")
		
		self.frame_predicting = LabelFrame (self,text="Predictions d'un article sur le web")
		self.frame_predicting.grid(row=1,padx=10,sticky="W")
		
		self.frame_testing = LabelFrame (self,text="Test sur un ensemble d'article")
		self.frame_testing.grid(row=2,pady=10,padx=10,sticky="W")
		
		# ========== Mise en place des widgets pour le training ============

		# Première ligne : dossier des articles d'entraînements
		self.label_training = Label(self.frame_training, text="Dossier des articles d'entraînements :")
		self.label_training.grid(row=0,column=0,sticky="W")
		
		self.dir_training = StringVar()
		self.entry_dir_training = Entry(self.frame_training, textvariable=self.dir_training,width=30)
		self.entry_dir_training.grid(row=0,column=1,padx=5)
		
		self.button_select_filename = self.button_training = Button (self.frame_training, text="Sélectionner un fichier de ce dossier", command=self.openfilename)
		self.button_select_filename.grid(row=0,column=2)
		
		self.dir_training = StringVar()
		self.entry_dir_training = Entry(self.frame_training, textvariable=self.dir_training,width=30)
		self.entry_dir_training.grid(row=0,column=1,padx=5)
		
		# Deuxième ligne : Nombre d'articles à prendre en compte
		self.label_nbr_articles = Label(self.frame_training, text="Nombres d'articles à prendre en compte :")
		self.label_nbr_articles.grid(row=1,column=0,sticky="W")
		
		self.nbr_articles_var = StringVar()
		self.entry_nbr_articles = Entry(self.frame_training, textvariable=self.nbr_articles_var,width=30)
		self.entry_nbr_articles.grid(row=1,column=1,padx=5)
		
		
		
		
		
		
		self.button_training = Button (self.frame_training, text="Entrainement", command=self.training).grid(row=2,column=1)
		
		# ======== Mise en place des widgets pour la prediction =========
	
		# Première ligne : insertion du lien
		self.label_predicting = Label(self.frame_predicting, text="Lien de l'article :")
		self.label_predicting.grid(row=0,column=0,sticky="W")
		
		self.link_article_var = StringVar()
		self.entry_link_article = Entry(self.frame_predicting,textvariable=self.link_article_var,width=50)
		self.entry_link_article.grid(row=0,column=1,padx=5,sticky="W")
		
		self.button_predicting = Button (self.frame_predicting, text="Prédiction", command=self.predicting)
		self.button_predicting.grid(row=0,column=2)
		
		# Deuxième ligne : affichage de la catégorie prédite
		self.label_category_text = Label (self.frame_predicting, text="Catégorie Prédite :")
		self.label_category_text.grid(row=1,column=0,sticky="WESN")
		
		self.category_var = StringVar()
		self.entry_category = Entry(self.frame_predicting,textvariable=self.category_var,width=20)
		self.entry_category.grid(row=1,column=1,padx=5,sticky="WNS")
		
		# ======= Mise en place des widgets pour le testing =========
		
		# Première ligne : nombre d'articles à tester
		self.label_nbr_articles_test = Label(self.frame_testing, text="Nombre d'articles à tester :")
		self.label_nbr_articles_test.grid(row=0,column=0,sticky="W")
		
		self.nbr_articles_test_var = StringVar()
		self.entry_nbr_articles_test=Entry(self.frame_testing,textvariable=self.nbr_articles_test_var,width=15)
		self.entry_nbr_articles_test.grid(row=0,column=1,padx=5,sticky="W")
		
		self.button_testing = Button (self.frame_testing, text="Tester", command=self.testing)
		self.button_testing.grid(row=0,column=2)
		
		# Deuxième ligne : affichage du score
		self.label_score_text = Label (self.frame_testing, text="Score :")
		self.label_score_text.grid(row=1,column=0,sticky="WESN")

		self.entry_score = Text(self.frame_testing,width=60,height=9)
		self.entry_score.grid(row=1,column=1,padx=5,pady=5,sticky="WNS")
	
	def openfilename(self):
		''' Gère la selection du dossier d'entraînement'''
		self.filename = filedialog.askopenfilename(initialdir = "/home/artix41/Bureau/TIPE/SVM/articles_train",title = "Choose your file")
		self.dirname = self.filename[:self.filename.rfind(filesep)]
		self.dir_training.set(self.dirname)
		
	def training(self):
		''' Gère la partie entraînement'''
		debut = time.time()
		print ("******************************************")
		print ("************** Training ******************")
		print ("******************************************\n")
		
		
		
		
		
		
		self.nbr_articles_train = int(self.nbr_articles_var.get())
		
		self.filenames = glob(os.path.join(self.dirname,'*'))
		filenames_train = self.filenames[:self.nbr_articles_train]
		category_train = [index_of_category((filename[filename.index('.')+1:])) for filename in filenames_train]
		
		# ======= Instantie le modèle de gestion du texte =======
		self.vectorizer = StemmedVectorizer(strip_accents="unicode")


		print ("========= Reading the training files ===========\n")
		
		corpus_train = [open(filename,"r").read() for filename in filenames_train]
		X_train = self.vectorizer.fit_transform(corpus_train)
		print ("[+] Success : reading train articles and vectorization\n")
		
		print ("======= Instantiate a predictive model =======\n")
		
		self.model = svm.LinearSVC()
		
		print ("=========== Fitting ================\n")
		
		self.model.fit(X_train.toarray(), category_train)
		print ("[+] Fitting success\n")
		
	def predicting(self):
		''' Gère la prédiction sur un article web'''
		self.url_article = self.link_article_var.get()
		
		print ("\n**********************************************")
		print ("************** Predicting ******************")
		print ("**********************************************\n")
		
		print ("========= Reading the article in the link =========\n")
	
		text_to_predict = url_to_text(self.url_article)
		
		print ("[+] Text opened succefully")
		
		print  ("========= Predicting =========\n")
		
		X_pred = self.vectorizer.transform([text_to_predict])
		category_pred = self.model.predict(X_pred.toarray())
		print ("[+] Predicting success\n")
		self.category_var.set(CATEGORY[category_pred])
		
	def testing(self):
		''' Gère la prédiction sur d'autres articles wikipedia'''
		nbr_articles_test = int(self.nbr_articles_test_var.get())
		
		filenames_test = self.filenames[self.nbr_articles_train+1:self.nbr_articles_train+1+nbr_articles_test]
		category_test = [index_of_category((filename[filename.index('.')+1:])) for filename in filenames_test]
		
		print ("========= Reading the testing files ===========\n")
		
		corpus_test = [open(filename,"r").read() for filename in filenames_test]
		X_test = self.vectorizer.transform(corpus_test)
		print ("[+] Success : reading testing files and vectorization\n")
		
		category_pred = self.model.predict(X_test.toarray())
		print ("[+] Predicting success\n")

		print ("Predictions : ")
		print (category_pred)
		print ("\nReality : ")
		print (np.array(category_test))
		
		print ("\n========= Scores =========\n")
		
		score = metrics.classification_report(category_test,category_pred)
		print (score)
		
		# Écriture du score
		self.entry_score = Text(self.frame_testing,width=60,height=9)
		self.entry_score.insert(END,score)
		self.entry_score.grid(row=1,column=1,padx=5,pady=5,sticky="WNS")







def main():
	window = Tk()
	interface = Interface(window)
	interface.mainloop()
if __name__ == "__main__":
	main()