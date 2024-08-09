import math

#calcola l'angolo tra i vettori ab e bc con b come veritce, a b c sono  coordinate (x,y)
def get_angle(a, b, c):
    # Calcola l'angolo (in radianti) del vettore ba rispetto all'asse x
    angle_ba = math.atan2(a[1] - b[1], a[0] - b[0])
    
    # Calcola l'angolo del vettore bc rispetto all'asse x
    angle_bc = math.atan2(c[1] - b[1], c[0] - b[0])
    
    # Calcola la differenza tra i due angoli e converto in in gradi 
    ang = math.degrees(angle_bc - angle_ba)
    
    # Se l'angolo è negativo, aggiungi 360 gradi per ottenere un angolo positivo equivalente
    if ang < 0:
        ang += 360
    
    # Restituisci l'angolo tra i due vettori
    return ang