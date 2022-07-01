Spiegazione dei file:
0. from_txt_to_pickle.py
                => trasformare il file glove.6B.300d.txt (GloVe) in formato pikle per una migliore gestione all-interno del DataLoader
                ==> glove.6B.300d.pickle

1. preprocessing1.py 
                => questo file serve a ridurre la dimensione del dataset intero eliminando le colonne non necessarie per permettere di caricarlo in una sola volta invece che a chunk (diventa cosí piú semplice da maneggiare) e messo tutto in un unico json invece di avere un file con tantissimi json
                ==> yelp_ridotto.json

2. preprocessing2.py
                => questo file serve a ridurre il numero di dati nel dataset per velocizzare l'apprendimento, prendendo da yelp_ridotto.json solo i buissness_id con almeno 150 reviews
                => yelp_ridotto2.json

3. preprocessing2.py
                => questo file serve a ridurre il numero di dati nel dataset per velocizzare l'apprendimento, prendendo da yelp_ridotto2.json solo i user?id con almeno 50 reviews
                => yelp_ridotto3.json
     
4. trainining.py
                => 
                ==> vari modelli nella cartella models

5.testing.py
		            => viene usato il miglior modello della cartella models (model_epoch_ 2.pth) per vedere la loss sul db di test
                ==> Test Loss: 

6.testing_lattn.py
		            => viene usato il miglior modello della cartella models (model_epoch_lattn_ 2.pth) per vedere la loss sul db di test
                ==> Test Loss: 

7.testing_gattn.py
		            => viene usato il miglior modello della cartella models (model_epoch_GAttn_ 5.pth) per vedere la loss sul db di test
                ==> Test Loss: 

*. utility.py 
		    => contiene una funzione per stampare a video una review con le parole colorate






