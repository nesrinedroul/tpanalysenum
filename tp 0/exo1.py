import matplotlib.pyplot as plt
import numpy as np
x= np.linspace(0, 10, 100)
y= np.sin(x)
xs= np.linspace(0, 10, 100)
ys=np.cos(xs)
plt.plot(x,y,'o-b',label='sin(x)') # trace la courbe y=sin(x) en bleu avec des croix
plt.plot(xs,ys,'o-r',label='cos(x)') # trace la courbe y=cos(x) en rouge avec des ronds
plt.title('representation des fonction trigonomic')
plt.xlabel('les x')
plt.ylabel('temps')
plt.legend() # affiche les légendes des courbes
plt.show() # affiche la figure a l'ecran
