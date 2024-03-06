# SylvaRoaD 
## Outil d'aide à la conception de nouvelles dessertes forestières 

Version : 0.2  
Langage : Python 3.x  

Financement : **Office National des Forêts - Pôle RDI de Chambéry** (*laurent.malabeux@onf.fr*)  
![ONF](./sylvaroad/img/onf_logo.gif?raw=true)

Conception et développement : **Sylvain DUPIRE, SylvaLab - 2021** (*sylvain.dupire@sylvalab.fr*)   
![SylvaLab](./sylvaroad/img/logo_sylvalab.png?raw=true)

---
### Contexte et objectif de SylvaRoaD

Les modèles numériques de terrain (MNT) dérivés de données spatiales à haute résolution (LiDAR, radar...) permettent aujourd'hui de représenter très fidèlement le relief. Cette représentation de qualité ouvre la porte à de nombreuses applications numériques pour faciliter le travail de terrain des forestiers, en particulier dans les zones les plus accidentées.

Dans le cadre de la conception de nouvelles dessertes forestières (route ou piste), ce type de MNT autorise l'utilisation d'algorithmes de recherche du meilleur itinéraire à la fois performants et précis.

SylvaRoaD s'inscrit dans cette logique et ambitionne d'aider les forestiers dans la conception de nouvelles dessertes forestières en leur permettant d'identifier un tracé optimal respectant plusieurs critères définis par l'utilisateur afin de desservir de nouvelles parcelles forestières. L'outil permet ainsi de tester rapidement plusieurs options d'itinéraires ou encore de rajouter des points de passages obligatoires. 
Les différents tracés ainsi identifiés peuvent être incorporés dans un GPS afin d'en valider la faisabilité sur le terrain.  

![SylvaLab](./sylvaroad/img/illustration.png?raw=true)

Tags = ___Desserte___, ___forestière___, ___montagne___, ___route___, ___MNT___, ___tracé___, ___conception___, ___création___, ___Forest___, ___Road___, ___Design___, ___DTM___  

---  
### Données en entrée de SylvaRoaD

**Données spatiales obligatoires** :

- Modèle numérique de terrain (raster de préférence issu de données Lidar)
- Points de passage obligatoires (shapefile de points) incluant le départ et l'arrivée de l'itinéraire à concevoir et (optionnellement) des points de passages intermédiaires.

**Données spatiales optionnelles** :

- Un dossier contenant des couches d'obstacles (tous formats acceptés)
- Une couche décrivant le foncier et permettant de discriminer les parcelles où le tracé peut passer de celles où ce n'est pas le cas

**Paramètres de modélisation** :

- Pente en travers maximale
- Pente en travers maximale pour un virage en épingle
- Angle au-delà duquel un virage est considéré comme épingle
- Pente en long minimale
- Pente en long maximale
- Pénalité de changement de direction*
- Pénalité de changement de pente*
- Taille du voisinage de recherche*
- Différence d’altitude maximale entre le tracé et le terrain
- Longueur maximale cumulée où pente en travers > Pente en travers maximale


*Paramètres inspirés du plugin Qgis Forest Road Designer (<https://github.com/GobiernoLaRioja/forestroaddesigner/>).


### Données en sortie de SylvaRoaD

- Document rappelant les paramètres de modélisation ainsi que les résultats par tronçon testé
- Un polyligne (shapefile) par tronçon optimal identifié par le processus 
- Un polyligne (shapefile) où les lacets sont retracés par tronçon optimal identifié par le processus 
    
&nbsp;   
  

  
---  


__Licence SylvaRoaD :__

    Copyright (C) 2021 by Sylvain DUPIRE - SylvaLab.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses/



