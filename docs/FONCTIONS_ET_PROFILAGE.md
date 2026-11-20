# Projet VOTAA : principales fonctions du code et profilage

Ce document décrit les fonctions de `main.cpp`, leur rôle, le choix qui est derrière chacune, et les endroits où le temps est mesuré (profilage).

Les numéros de ligne correspondent au `main.cpp` de l'archive finale (949 lignes).

---

## 1. Vue d'ensemble

À chaque image de la webcam, le programme fait toujours la même boucle :

```text
webcam.read  ->  resize 640x480
     |
     v
detecterVoitures(scene)          TACHE 1 : quelles photos, où sont leurs 4 coins
     |
     v
convertirDetections(...)         lien entre le format de la tâche 1 et celui de la tâche 2
     |
     v
augmenterScene(...)              TACHE 2 : une vidéo incrustée par photo
     |
     v
affichage (imshow) + mesures     temps par étape, FPS réel, précision
```

Le fichier est découpé en deux sections : `TACHE 1` (lignes 27 à 287) puis `TACHE 2` (lignes 289 à 949, avec le `main`).

---

## 2. Les structures de données

| Structure | Ligne | Contenu | Pourquoi |
|---|---|---|---|
| `Detection` | 29 | nom de la voiture, 4 coins, homographie, mode (suivi ou détection), nombre d'inliers | interface fixée au début entre les deux tâches : chacun a pu développer seul |
| `Reference` | 291 | une photo, sa vidéo, un `VideoCapture`, l'état de lecture | un lecteur par photo : chaque vidéo garde sa propre position |
| `ObjetDetecte` | 304 | numéro de la photo, 4 coins, mode, nombre d'inliers | format de travail de la tâche 2, rempli à partir de `Detection` |
| `Mesures` | 314 | durée de chaque étape d'une image, en ms | sert au bilan de temps (section 6) |
| `Evaluation` | 659 | photos déclarées visibles, vrais positifs, faux positifs, manqués | sert au calcul de la précision |

---

## 3. Tâche 1 : détection et suivi

| Fonction | Ligne | Ce qu'elle fait | Pourquoi ce choix |
|---|---|---|---|
| `lirePhoto` | 71 | lit une photo de référence, gère la transparence (PNG à 4 canaux), la réduit à 400 px | les photos vont de 596 à 3 748 px de large : une taille commune rend la détection plus rapide et plus régulière |
| `chargerReferences` | 100 | calcule les points AKAZE des 12 photos (`detectAndCompute`) | fait **une seule fois au démarrage** : on ne recalcule jamais les points des photos de référence |
| `akazeTracker` | 134 | apparie la scène avec une photo (`knnMatch`, k = 2, ratio test 0,8), calcule l'homographie (`findHomography` + RANSAC), projette les 4 coins (`perspectiveTransform`) | le ratio test élimine les appariements ambigus ; RANSAC élimine les faux appariements restants ; la photo n'est acceptée qu'avec au moins 12 inliers et un cadre convexe d'au moins 1 000 px² |
| `suivreVoiture` | 184 | suit les points d'une photo d'une image à la suivante (`calcOpticalFlowPyrLK`), avec un aller-retour | le flot optique est beaucoup plus rapide qu'AKAZE ; l'aller-retour supprime les points qui dérivent (écart de plus de 3 px) ; moins de 8 points = photo perdue |
| `detecterVoitures` | 227 | fonction appelée par la tâche 2 : suit les photos déjà trouvées, relance AKAZE si rien n'est suivi ou toutes les 5 images, renvoie la liste | **stratégie hybride** : AKAZE est précis mais coûteux, le suivi est rapide ; AKAZE une image sur 5 suffit à trouver les nouvelles photos et à corriger la dérive |

Deux lignes de cette section ont été changées à l'intégration, pour le temps de calcul (voir section 6) :

- **ligne 152** : RANSAC n'est lancé que s'il y a au moins 12 bons appariements ;
- **ligne 157** (constante ligne 61) : RANSAC limité à 500 itérations au lieu de 2 000.

---

## 4. Tâche 2 : association, lecture et incrustation

### Association photo / vidéo

| Fonction | Ligne | Ce qu'elle fait | Pourquoi |
|---|---|---|---|
| `extraireMarque` | 346 | `"10-Range-Rover.mp4"` → `"Range-Rover"` | la marque est la seule clé fiable |
| `chargerVideos` | 360 | associe chaque photo à sa vidéo **par le nom de marque** et ouvre un `VideoCapture` par photo | les numéros 10 et 11 sont inversés entre les images et les vidéos : associer par numéro donnerait la mauvaise vidéo à Citroën et Range Rover |
| `trouverReference` | 467 | renvoie la position d'une photo dans la liste à partir de son numéro | le numéro de la photo n'est pas sa position dans la liste |
| `convertirDetections` | 479 | transforme chaque `Detection` en `ObjetDetecte`, en associant par nom | `Detection::id` est un indice de chargement (0 à 11), pas le numéro du fichier |

### Lecture de la vidéo

| Fonction | Ligne | Ce qu'elle fait | Pourquoi |
|---|---|---|---|
| `lireImageEnBoucle` | 407 | lit l'image suivante, revient au début en fin de vidéo | les vidéos durent 25 à 30 s : la démonstration peut durer plus longtemps |
| `lireImageSynchronisee` | 418 | fait avancer la vidéo selon le **temps réel écoulé** : saute des images avec `grab()` si le programme est lent, réutilise l'image courante s'il est rapide, met en pause si la photo a disparu plus de 0,5 s | sans cela, la vidéo ralentissait dès que le FPS baissait (une image vidéo par image webcam) |

### Recalage et composition

| Fonction | Ligne | Ce qu'elle fait | Pourquoi |
|---|---|---|---|
| `coinsImage` | 513 | les 4 coins de l'image vidéo, dans l'ordre haut gauche, haut droit, bas droit, bas gauche | même ordre que les coins donnés par la tâche 1 ; un ordre différent retournerait la vidéo |
| `quadrilatereValide` | 547 | vérifie 4 points finis, un quadrilatère convexe et une aire d'au moins 500 px² | une détection aberrante ne doit jamais déclencher une incrustation |
| `incrusterVideo` | 562 | `getPerspectiveTransform` (coins de la vidéo → coins de la photo), puis `warpPerspective`, puis masque `fillPoly` et `copyTo` | **4 correspondances exactes** : `getPerspectiveTransform` suffit, RANSAC est inutile ici. Le masque vient de la géométrie, pas des couleurs : rien ne déborde et les zones noires de la vidéo restent visibles. `BORDER_REPLICATE` évite un liseré noir au bord |
| `augmenterScene` | 615 | boucle sur tous les objets : vérifie, lit la vidéo de chacun, incruste, dessine | c'est le multi-objets : chaque photo a sa vidéo, et un objet invalide ou inconnu est ignoré |
| `dessinerObjet` | 598 | contour vert, 4 sommets rouges, étiquette « numéro - marque \| AKAZE / Tracking » | rend visibles la détection et le suivi pendant la démonstration |
| `simulerObjets` | 521 | 1 à 3 objets à des positions fixes, sans détection | tester la tâche 2 seule, et mesurer son coût sans la détection |

---

## 5. Évaluation de la précision

| Fonction | Ligne | Ce qu'elle fait |
|---|---|---|
| `saisirEvaluation` | 691 | après la touche `e`, demande dans la console les numéros des photos réellement visibles : c'est la vérité terrain |
| `compterImage` | 715 | pour chaque image : photo déclarée et détectée = vrai positif ; détectée mais non déclarée = faux positif ; déclarée mais non détectée = manquée |
| `precision`, `rappel` | 669, 675 | précision = correctes / (correctes + fausses) ; rappel = correctes / (correctes + manquées) |
| `afficherBilanPrecision` | 731 | tableau par séquence de 300 images, puis le total |

**Pourquoi ce protocole :** le sujet demande la précision, et elle ne peut pas se déduire du temps de calcul. Il faut savoir ce qui est réellement devant la caméra ; l'opérateur le déclare. Limite : on vérifie qu'on reconnaît la bonne photo, pas la position exacte des coins.

---

## 6. Où est fait le profilage

Le profilage se fait à **deux niveaux**, avec le même outil : `getTickCount()` au début et à la fin d'un bloc, divisé par `getTickFrequency()` pour obtenir des millisecondes.

### Niveau 1 : le temps de chaque étape d'une image (tâche 2)

Il répond à la question : **où passe le temps dans une image ?**

| Étape mesurée | Où dans le code | Fonction de calcul |
|---|---|---|
| capture | `main`, lignes 864 à 873 | `webcam.read` + `resize` |
| détection | `main`, lignes 878 à 880 | `detecterVoitures` |
| lecture vidéo | `augmenterScene`, lignes 629 à 631 | `lireImageSynchronisee` |
| projection | `incrusterVideo`, lignes 568 à 574 | `getPerspectiveTransform` + `warpPerspective` |
| composition | `incrusterVideo`, lignes 576 à 588 | `fillPoly` + `copyTo` |
| affichage | `main`, lignes 894 à 904 | texte + `imshow` + `waitKey` |
| boucle complète | `main`, lignes 861 et 905 | sert au FPS réel |

- La fonction de mesure est `millisecondesDepuis` (ligne 328) ; les durées sont rangées dans la structure `Mesures` (ligne 314).
- Le **FPS réel** = images traitées / durée totale des boucles, capture et affichage compris (lignes 914 et 915). Ce n'est pas 1 000 / temps de calcul.
- Les 30 premières images après un changement de mode sont ignorées (échauffement, ligne 845).
- Le bilan s'affiche à la fermeture : `afficherBilan` (ligne 764) donne la moyenne de chaque étape, puis la moyenne, la médiane, le p95 et le maximum du temps de traitement, par nombre de photos visibles.

**Ce qu'il a montré :** l'incrustation coûte 4 à 5 ms par photo ; la détection 43 à 50 ms en moyenne. C'est la détection qui limite le programme.

### Niveau 2 : l'intérieur de la détection (dans la section tâche 1)

Il répond à la question suivante : **dans la détection, quelle étape coûte cher ?** Ces lignes ont été ajoutées dans le code de la tâche 1 et ne changent aucun résultat.

| Étape mesurée | Où dans le code | Compteur |
|---|---|---|
| extraction AKAZE de la scène | `detecterVoitures`, lignes 249 à 253 | `tempsAkaze`, `pointsScene`, `imagesDetection` |
| appariement avec chaque photo | `akazeTracker`, lignes 140 à 142 | `tempsAppariement` |
| homographie RANSAC | `akazeTracker`, lignes 156 à 159 | `tempsRansac`, `appelsRansac` |
| suivi par flot optique | `detecterVoitures`, lignes 233 à 241 | `tempsSuivi` |

- Les compteurs sont déclarés lignes 64 à 66, avec la petite fonction de mesure `msDepuis` (ligne 68).
- Le résultat s'affiche à la fermeture : `afficherProfilDetection` (ligne 793), avec le temps moyen de chaque étape par image de détection, sa part en %, le nombre de points de la scène, le nombre d'appels RANSAC et le nombre de cœurs du processeur.

**Ce qu'il a montré :** sur une image de détection, RANSAC prenait environ 70 % du temps. Il était lancé pour les 12 photos, même absentes, et pour chacune il allait jusqu'à 2 000 itérations sans rien trouver. D'où les deux changements des lignes 152 et 157.

| Mesure, une photo | Avant | Après |
|---|---|---|
| RANSAC par image de détection | 91,7 ms (12 appels) | 14,7 ms (2 appels) |
| Précision | 100 % | 100 % |

Ce gain a été mesuré dans un essai où l'image contenait peu de points (173). Quand la scène en contient plus de 300, davantage de photos dépassent le seuil de 12 appariements et RANSAC redevient le poste principal : c'est la piste d'amélioration suivante (paralléliser les 12 homographies).

---

## 7. Les constantes à connaître

| Constante | Valeur | Ligne | Rôle |
|---|---|---|---|
| `MIN_INLIERS` | 12 | 59 | inliers minimum pour accepter une photo |
| `PERIODE_DETECTION` | 5 | 60 | AKAZE relancé toutes les 5 images |
| `ITERATIONS_RANSAC` | 500 | 61 | limite d'itérations de RANSAC à la détection |
| `pauseMaximale` | 0,5 s | 420 | au-delà, la vidéo était en pause |
| `sautMaximal` | 15 images | 421 | rattrapage maximal de la vidéo après une image lente |
| `imagesEchauffement` | 30 | 845 | images ignorées dans les mesures après un changement de mode |
| `imagesEvaluation` | 300 | 846 | longueur d'une séquence d'évaluation de la précision |
