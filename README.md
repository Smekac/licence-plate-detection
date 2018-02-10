# licence-plate-detection
Soft computing project

Pokretanje pograma:
1. Pokretanje programa se vrši tako što se pokrenefajl plate-isolation.py sa kojim popunimo folder "ImagesFrame" sa frejmovima(slikama) iz videa. 

2. Zatim se pokrene fajl slikaKolaTablice.py gde se vrši detekcija tablica samo sa par prosledjenih frejmova (zbog memorije i zbog lošeg snimka gde su tablice veoma mutne) i detektovane tablice kao sliku png formata snimamo u folder "plates".

3. Da bi imali više slika tablica za detekciju i zbog lošeg kvaliteta frejma, pokreće se detectionPlateForPictures.py
koji detektuje tablice sa slika vozila iz foldera "ImagesOfCars"

4. Za pronalaženje karaktera sa tablice potrebno je pokrenuti fajl char-isolation.py i u njemu postaviti željenu fotografiju u promenljivoj allPlates. Fotografije tablica se nalaze u folderu "plates". 

Članovi tima :
<p> •	<b>Ivan Vukašinović RA53/2014 </b> </p>
<p> •	<b>Dejan Stojkić RA 177/2014 </b> </p>

Problem koji se rešava?
Detekcija i prepoznavanje registarskih tablica automobila na video snimku u realnom vemenu . Prvi problem jeste detekcija vozila i izdvajanje tabilca sa registrovanih objekata (vozila) za dati snimak. Zatim kada je tabilica detektovana izdvajaju se karakteri sa date tablice .
Rešenje može biti korišćeno u razne svrhe gde je potrebna automatska detekcija tablica (npr. kod parking servisa (kamere na ulazu pored rampe) , naplatne rampe na autoputu , za potrebe policije detektovanja tablica vozila ).

Algoritmi koji će se koristiti?
Adaptivni thresholding za filtriranje slike . Izdvajanje dela na kojem se nalazi tablica i njegova obrada za konvolucionu neuronsku mrežu (OCR) .
Koristili bismo OpenCV biblioteku na Python programskom jeziku i njene mogućnosti u real time režimu. ( https://www.youtube.com/watch?v=iS_yuSEFXxM slicno )

Metrika za poređenje performansi algoritama :
Metrika će biti preciznost tačno prepoznatih (karaktera) tablica na osnovu tačnih vrednosti tablica iz određenog fajla u kome se nalaze podaci o tablicama datih vozila. Broj tačno prepoznatih tablica, porediće se rešenje sa fajlom koji sadrži podatke na osnovu videa.

Podaci koji se koriste:
Video snimak koji ćemo koristiti pokušaćemo ručno snimiti. Koristićemo dataset za obučavanje neuronske mreže koji se sastoji iz karaktera.

Validacija rešenja
Broj tačno prepoznatih tablica, porediće se rešenje sa fajlom koji sadrži podatke na osnovu videa.

Repozitorijum: https://github.com/ivukasinovic/licence-plate-detection
Asistent: Ivan Perić @ivan7792
