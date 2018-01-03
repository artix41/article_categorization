from sys import argv
import os
import urllib.request as urllib
import urllib.parse as parse
from bs4 import BeautifulSoup
import time

if __name__ == '__main__':
	# Lecture de la liste des articles wiki à considérer
	try:
		filename = argv[1]
		category = argv[2]
	except:
		print ("Error : 2 arguments excepted")
		print ("python3 wiki_to_text.py <file_with_the_links> <category> [<id first article>]")
		print ("The text files will be named 'number.category'")
		exit(0)
	if len(argv) >= 4:
		id_article = int(argv[3])
	else:
		id_article = 0
	try:
		with open (filename,"r") as f:
			list_url = f.readlines()
	except:
		print ("The file given doesn't exist")
		exit(0)
		
	list_url = [line.strip() for line in list_url if line != '' and line != '\n' and line != ' ']
	list_url = [list_url[i] for i in range(0,len(list_url))]
	
	for url in list_url:
		if url[-4:] != ".jpg" and url[-4] != ".svg": 
			try:
				# Décodage pour les accents
				url = url.split("wiki/")
				url[1] = parse.quote(parse.unquote(url[1]))
				url = "wiki/".join(url)
				
				html = urllib.urlopen(url).read()
				soup = BeautifulSoup(html)

				# kill all script and style elements
				for script in soup(["script", "style"]):
					script.extract()    # rip it out

				# get text
				text = soup.get_text()

				# break into lines and remove leading and trailing space on each
				lines = (line.strip() for line in text.splitlines())
				# break multi-headlines into a line each
				chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
				# drop blank lines
				text = '\n'.join(chunk for chunk in chunks if chunk)
				
				with open (os.path.join("articles_train",str(id_article).zfill(3) + "." + category),"w") as file:
					file.write(text)
				print ("[+] " + str(id_article) + " Success : " + url)
				id_article += 1
			except KeyboardInterrupt:
				exit(0)
			except:
				print ("[-] Fail : " + str(url))
			
	
