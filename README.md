# VOTAA : vision augmentée et recalage dynamique 2D multi-objets en temps réel

Projet de vision par ordinateur (Master 2), réalisé en binôme par **Mélissa KHOUDI** et **Haitam EL ARAR**, soutenu le 18 novembre 2026.

Le programme reconnaît des photos de voitures dans le flux d'une webcam, les suit, et incruste sur chacune la vidéo de sa marque, en perspective. Plusieurs photos peuvent être augmentées en même temps.

**Langage et outils :** C++, OpenCV 4.5.4, Visual Studio 2022 (Release x64).

---

## Principe

```text
Webcam (640x480)
      |
      v
Tâche 1 : detecterVoitures   AKAZE + ratio test + homographie RANSAC, puis suivi Lucas-Kanade
      |                      -> nom de chaque photo visible + ses 4 coins
      v
Tâche 2 : augmenterScene     getPerspectiveTransform + warpPerspective + masque + copyTo
      |                      -> une vidéo incrustée par photo
      v
Affichage + mesures          temps par étape, FPS réel, précision et rappel
```

| Tâche | Responsable | Contenu |
|---|---|---|
| 1 : détection et suivi | Mélissa KHOUDI | reconnaissance AKAZE, homographie RANSAC, suivi par flot optique, plusieurs voitures à la fois |
| 2 : incrustation et mesures | Haitam EL ARAR | association photo et vidéo, lecture synchronisée, incrustation en perspective, mesures de temps et de précision, intégration |

Le détail de chaque fonction, et l'endroit où le temps est mesuré, sont dans [docs/FONCTIONS_ET_PROFILAGE.md](docs/FONCTIONS_ET_PROFILAGE.md).

---

## Contenu du dépôt

| Élément | Contenu |
|---|---|
| `main.cpp` | tout le programme, en deux sections : `TACHE 1` puis `TACHE 2` et le `main` |
| `Reference Images-20261111/` | les 12 photos de référence |
| `Overlay Videos-20261111/` | les 12 vidéos à incruster (1280×720, 30 images/s) |
| `docs/FONCTIONS_ET_PROFILAGE.md` | les fonctions principales, leurs choix, et le profilage |

Les photos et vidéos ont été fournies dans le cadre du cours.

---

## Compiler et lancer

1. Créer un projet console C++ dans Visual Studio 2022 et y ajouter `main.cpp`.
2. En **Release x64**, dans les propriétés du projet :
   - C/C++ > Répertoires Include supplémentaires : `<OpenCV>\include`
   - Éditeur de liens > Répertoires de bibliothèques supplémentaires : `<OpenCV>\lib`
   - Éditeur de liens > Entrée > Dépendances supplémentaires :
     `opencv_core454.lib; opencv_imgcodecs454.lib; opencv_features2d454.lib; opencv_imgproc454.lib; opencv_highgui454.lib; opencv_videoio454.lib; opencv_calib3d454.lib; opencv_video454.lib`
   - Débogage > Environnement : `PATH=<OpenCV>\bin;%PATH%`, et Répertoire de travail : `$(ProjectDir)`
3. Placer les deux dossiers de données à côté de `main.cpp`.
4. Lancer avec Ctrl+F5.

Les dossiers peuvent porter d'autres noms : `Project_final.exe <dossier_images> <dossier_videos>`.

### Touches

| Touche | Effet |
|---|---|
| `d` | détection réelle (au démarrage) |
| `s` | objets simulés, sans détection (tâche 2 seule) ; `0` à `3` pour leur nombre |
| `e` | évaluer la précision : taper dans la console les numéros des photos visibles |
| `Échap` | quitter et afficher les bilans : temps par étape, profil de la détection, précision |

---

## Résultats

Mesures du 16 novembre 2026, photos 3, 4 et 5 affichées sur un téléphone, processeur à 12 cœurs logiques.

| Photos visibles | Détection | Incrustation | FPS réel |
|---|---|---|---|
| 1 | 47,0 ms | 4,2 ms | 14,0 |
| 2 | 42,9 ms | 8,8 ms | 14,1 |
| 3 | 49,6 ms | 14,9 ms | 12,0 |

- **Tâche 2 seule** (objets simulés) : 4 à 5 ms par objet, 30 images/s, la limite de la webcam.
- **Précision : 100 %** (1 798 détections sur 1 798, aucune fausse). **Rappel : 99,9 %** (1 798 sur 1 800).
- Le FPS réel compte toute la boucle, capture et affichage compris.

### Optimisation de la détection

Le profilage a montré que RANSAC prenait environ 70 % du temps d'une image de détection : il était lancé pour les 12 photos, même absentes. Deux changements, sans perte de précision :

- RANSAC n'est lancé qu'à partir de 12 bons appariements, le minimum pour valider une photo ;
- RANSAC est limité à 500 itérations au lieu de 2 000.

Dans un essai avec une scène pauvre en points, la détection est passée de 58 à 17 ms et le programme de 12 à 22 images/s. Le gain est plus faible quand la scène contient beaucoup de points.

---

## Limites

- Le temps réel (30 images/s) n'est pas atteint avec la détection : une image sur cinq relance AKAZE et prend 165 à 210 ms.
- Testé jusqu'à 3 photos simultanées.
- La précision est évaluée sur l'identification des photos, pas sur la position exacte des coins ; la vérité terrain est déclarée par l'opérateur.
- Deux exemplaires de la même photo ne sont pas gérés.

## Pistes

- Paralléliser les 12 homographies sur les cœurs du processeur.
- Faire la détection dans un fil séparé du suivi.
- Présélectionner les zones avec YOLO ; incruster des objets 3D avec `solvePnP`.
