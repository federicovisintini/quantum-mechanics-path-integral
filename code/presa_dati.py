from particella_cerchio import *
import time

# importo l'intero file particella_cerchio.py (mettere le simulazioni in fondo rompeva molto la leggibilità del documento)


# PARAMETRI

# Parametri sistema fisico
# 1. chi: frequenza tipica problema (cioè chi = h_plank/(4*np.pi**2 * m * R**2) )
# 2. beta: temperatura inversa
# 3. N: numero di divisioni temporali
# 4. a = beta * h_plank / N: spacing temporale (non indipendente)
# 5. eta = a * chi: rapporto fra spacing temporale e tempo caratteristico sistema (non indipendente)

# In virtù di queste definizioni scelgo di studiare il problema con le seguenti quantità indipendenti:
N = 200
beta = 4  # beta*h*chi
chi = 1  # fissa le dimensioni (non serve come variabile ?)
eta = beta / N  # a in unità adimensionali
# notiamo per inciso che il prodotto N * eta = beta * (h_plank * chi)
# il limite al continuo di fa per N grande e N*eta=costante (a T fissata)

# Parametri legati alla simulazione (quanti step fare etc.)
misure = int(1e6) # numero di misure da prendere
decorrel = 3  # numero di metropolis fra le misure
termalizzazione = int(1e4)  # quanti passi butto perchè stavo termalizzando
# parametro per cui voglio accettanza ca 0.4/0.6
#delta_metr = 2 * np.sqrt(eta)
delta_metr = 0.5

# Definisco gli N che voglio studiare in critical slowing down
N_list = list(range(50, 500, 50))  # range è un iterabile
N_list_taylor = list(range(50, 500, 50))  # range è un iterabile


## SIMULAZIONE (PARTE 1) - COS'E' il WINDING NUMBER
# calcolo il tempo necessario per questa simulazione
start = time.time()

print('Parte 1:')

# inizializzo vettore y 'a freddo'
y = np.zeros(N)

# termalizzazione
for i in range(termalizzazione):
    update_metropolis(y, eta, delta_metr)

# presa misure (misuro Q: winding number)
measures = np.ndarray(misure)

# Creo una barra di avanzamento
# prog = pyprog.ProgressBar("", "", misure)
# Update Progress Bar
# prog.update()


# prendo 'misure' misure ogni 'decorrel' updates del reticolo
for i in range(misure):
    for j in range(decorrel):
        update_metropolis(y, eta, delta_metr)
    measures[i] = winding_number(y)

    # Set Progress Bar current status and update
    # prog.set_stat(i + 1)
    # prog.update()
    if 100 * i % misure == 0:
        print(int(100 * i / misure),'% completato')


# Make the Progress Bar final
# prog.end()

# salvo la simulazione su file
simul_path = 'DATA/winding_number.dat'
with open(simul_path, 'w') as file:
    # innanzitutto mi scrivo chi sono beta, chi e N
    file.write("%f\n" % beta)
    file.write("%d\n" % chi)
    file.write("%d\n" % N)

    # ora mi scrivo le misure del winding number (array lungo misure)
    for Q in measures:
        file.write("%d " % Q)
    file.write("\n")

    # infine salvo l'ultima configurazione del campo
    for phi in y:
        file.write("%f " % phi)


## SIMULAZIONE (PARTE 2) - CRITICAL SLOWING DOWN
print('\nParte 2 (critical slowing down):')
# ripeto la simulazione della parte 1, ma variando N, mostrando il critical slowing down
# nel limite al continuo (T fissa), è più difficile cambiare Q

# salvo i risultati su file
simul_path = 'DATA/critical_slowing_down.dat'
with open(simul_path, 'w') as file:
    # inanzitutto mi scrivo chi sono beta e chi
    file.write("%f\n" % beta)
    file.write("%d\n" % chi)

    # poi mi scrivo N e la MC di Q associate a questo N
    # innanzittutto N
    for N in N_list:
        file.write("%d " % N)
    file.write("\n")

    # Creo una barra di avanzamento
    # prog = pyprog.ProgressBar("", "", len(N_list))
    # Update Progress Bar
    # prog.update()

    # ora faccio la simulazione e la salvo passo passo
    for N in N_list:
        # imposto i parametri della simulazione
        # beta = 5
        eta = beta / N
        #delta_metr = 0.5  # fissiamo delta a 0.5 per non avere troppa autocorrelazione
        #delta_metr = 2 * np.sqrt(eta)

        # inizializzo a freddo
        y = np.zeros(N)

        # termalizzazione
        for i in range(termalizzazione):
            update_metropolis(y, eta, delta_metr)

        # inizializzo l'array delle misure lungo quanto mi serve, non di più
        measures = np.ndarray(misure)

        # misuro il winding number
        for i in range(misure):
            # facendo decorrel passi fra le misure
            for j in range(decorrel):
                update_metropolis(y, eta, delta_metr)
            measures[i] = winding_number(y)

        # ora scrivo le misure del winding number relativo ad N (array lungo misure)
        for Q in measures:
            file.write("%d " % Q)
        file.write("\n")

        # Set Progress Bar current status and update
        # prog.set_stat(N_list.index(N) + 1)
        # prog.update()
        print(int(100 * N_list.index(N) / len(N_list)),'% completato')

    # Make the Progress Bar final
    # prog.end()


## SIMULAZIONE (PARTE 3) - TAYLOR METHOD
# ripeto la parte 2 (presa dati a vari N), usando taylor method
# mi calcolo il tempo di autocorrelazione della suscettibilità
print('\nParte 3 (taylor method):')

# salvo i risultati su file
simul_path = 'DATA/taylor.dat'
with open(simul_path, 'w') as file:
    # inanzitutto mi scrivo chi sono beta e chi
    file.write("%f\n" % beta)
    file.write("%d\n" % chi)

    # poi mi scrivo N e la MC di Q associate a questo N
    # innanzittutto N
    for N in N_list:
        file.write("%d " % N)
    file.write("\n")

    # Creo una barra di avanzamento
    # prog = pyprog.ProgressBar("", "", len(N_list_taylor))
    # Update Progress Bar
    # prog.update()

    # ora faccio la simulazione e la salvo passo passo
    for N in N_list_taylor:
        # imposto i parametri della simulazione
        # beta = 5
        eta = beta / N
        delta_metr = 0.5  # fissiamo delta a 0.5 per non avere troppa autocorrelazione
        # delta_metr = 2 * np.sqrt(eta)

        # inizializzo a freddo
        y = np.zeros(N)

        # termalizzazione
        for i in range(termalizzazione):
            update_metropolis(y, eta, delta_metr)

        # inizializzo l'array delle misure lungo quanto mi serve, non di più
        measures = np.ndarray(misure)

        # misuro il winding number
        for i in range(misure):
            # facendo decorrel passi fra le misure
            for j in range(decorrel):
                update_metropolis(y, eta, delta_metr)
                taylor_move(y, eta, 0.2 * delta_metr)
            measures[i] = winding_number(y)

        # ora mi scrivo le misure del winding number relativo ad N (array lungo misure)
        for Q in measures:
            file.write("%d " % Q)
        file.write("\n")

        # Set Progress Bar current status and update
        # prog.set_stat(N_list_taylor.index(N) + 1)
        # prog.update()
        print(int(100 * N_list_taylor.index(N) / len(N_list_taylor)),'% completato')

    # Make the Progress Bar final
    # prog.end()

end = time.time()
print('tempo =', end - start)