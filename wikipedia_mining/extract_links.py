''' ============== Wikipedia Link extractor ===================
Read a bunch of wiki html code in the file in argv[1] in the directory of the script and add all the links in the file in argv[2]
'''

import os
from sys import argv

try:
	text_file = argv[1]
	links_file = argv[2]
except:
	print ("Error : the script except 2 arguments")
	print ("python3 extract_links.py <file_with_html> <file_to_add_the_links>")
	exit(0)
	
with open(text_file,"r") as f:
	texte = f.read()
	texte = texte.split("<a href=\"")
	links = [texte[i].split("\"") for i in range (len(texte))]
	links = [links[i][0] for i in range (len(links))]
	links = ["http://fr.wikipedia.org" + links[i] for i in range (len(links))]
with open("links","w") as f:
	for i in range(len(links)):
		f.write(links[i])
		if i != len(links)-1:
			f.write("\n")

os.system("cat links >>" + links_file)
os.system ("rm -f links")
