# licence-plate-detection
Soft computing project

Pokretanje pograma:
Za pronalaženje karaktera sa tablice potrebno je pokrenuti fajl char-isolation.py i u njemu postaviti željenu fotografiju u promenljivoj allPlates. Fotografije tablica se nalaze u folderu "plates". 



Članovi tima :
•	Ivan Vukašinović RA53/2014
•	Dejan Stojkić RA 177/2014

Problem koji se rešava?
Detekcija i prepoznavanje registarskih tablica automobila na video snimku u realnom vemenu . Prvi problem jeste detekcija vozila i izdvajanje tabilca sa registrovanih objekata (vozila) za dati snimak. Zatim kada je tabilica detektovana izdvajaju se karakteri sa date tablice .
Rešenje može biti korišćeno u razne svrhe gde je potrebna automatska detekcija tablica (npr. kod parking servisa (kamere na ulazu pored rampe) , naplatne rampe na autoputu , za potrebe policije detektovanja tablica vozila ).

Algoritmi koji će se koristiti?
Adaptivni thresholding za filtriranje slike . Izdvajanje dela na kojem se nalazi tablica i njegova obrada za konvolucionu neuronsku mrežu (OCR) .
Koristili bismo OpenCV biblioteku na Python programskom jeziku i njene mogućnosti u real time režimu. ( https://www.youtube.com/watch?v=iS_yuSEFXxM slicno )

Metrika za poređenje performansi algoritama :
Metrika će biti preciznost tačno prepoznatih (karaktera) tablica na osnovu tačnih vrednosti tablica iz određenog fajla u kome se nalaze podaci o tablicama datih vozila. Broj tačno prepoznatih tablica, porediće se rešenje sa fajlom koji sadrži podatke na osnovu videa.

Podaci koji se koriste:
Video snimak koji samo napravili ili je preuzet sa interneta. Koristićemo dataset za obučavanje neuronske mreže koji se sastoji iz karaktera.

Validacija rešenja
Broj tačno prepoznatih tablica, porediće se rešenje sa fajlom koji sadrži podatke na osnovu videa.

Repozitorijum: https://github.com/ivukasinovic/licence-plate-detection
Asistent: Ivan Perić @ivan7792
