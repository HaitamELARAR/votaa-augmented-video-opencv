// Projet VOTAA - Vision augmentee et recalage dynamique 2D multi-objets en temps reel
// Touches : d = detection reelle, s = objets simules (0 a 3), e = evaluer la precision, Echap = quitter + bilans
// Donnees : dossiers ci-dessous, ou deux arguments (dossier_images dossier_videos)

// Dependances OpenCV 4.5.4:
//   Editeur de liens > Entree > Dependances supplementaires :
//     opencv_core454.lib;opencv_imgcodecs454.lib;opencv_features2d454.lib;opencv_imgproc454.lib;
//     opencv_highgui454.lib;opencv_videoio454.lib;opencv_calib3d454.lib;opencv_video454.lib


#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

string dossierImages = "Reference Images-20261111";
string dossierVideos = "Overlay Videos-20261111";

//  TACHE 1 : detection + tracking

struct Detection {
	int id;                    
	string nom;                
	Mat H;                     
	vector<Point2f> coins;    
	bool tracking;             
	int nbInliers;             
};

// photos de reference
vector<Mat> trainImages;
vector<string> noms;
vector<vector<KeyPoint>> kp_train;
vector<Mat> desc_train;
vector<vector<Point2f>> objectCorners;

Ptr<AKAZE> akaze;
Ptr<DescriptorMatcher> matcher;

// etat de chaque voiture
vector<bool> trackingActif;
vector<vector<Point2f>> sceneCorners;
vector<vector<Point2f>> ptsRef;      // points sur la photo
vector<vector<Point2f>> ptsScene;    // les memes dans l'image camera
vector<int> inliers;
Mat prevGray;
int numImage = 0;

const int TAILLE_REF = 400;      
const float RATIO = 0.8f;        
const int MIN_INLIERS = 12;       
const int PERIODE_DETECTION = 5;  
const int ITERATIONS_RANSAC = 500; // limite de RANSAC a la detection : optimisation du temps

//  profilage : temps cumules (ms) des etapes de la detection et du suivi
double tempsAkaze = 0, tempsAppariement = 0, tempsRansac = 0, tempsSuivi = 0;
long pointsScene = 0;
int imagesDetection = 0, appelsRansac = 0;

double msDepuis(int64 debut) { return (getTickCount() - debut) * 1000.0 / getTickFrequency(); }

// charge une photo (fond blanc si png transparent) et la redimensionne
Mat lirePhoto(const string& chemin) {
	Mat img = imread(chemin, IMREAD_UNCHANGED);
	if (img.empty()) return img;
	if (img.channels() == 4) {
		Mat fond(img.size(), CV_8UC3, Scalar(255, 255, 255));
		for (int y = 0; y < img.rows; y++)
			for (int x = 0; x < img.cols; x++) {
				Vec4b p = img.at<Vec4b>(y, x);
				float a = p[3] / 255.0f;
				for (int c = 0; c < 3; c++)
					fond.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(p[c] * a + 255 * (1 - a));
			}
		img = fond;
	}
	double echelle = (double)TAILLE_REF / max(img.cols, img.rows);
	resize(img, img, Size(), echelle, echelle, INTER_AREA);
	return img;
}

// "Reference_Images/1-Toyota.png" -> "Toyota"
string nomDepuisFichier(string f) {
	f = f.substr(f.find_last_of("/\\") + 1);
	f = f.substr(0, f.find_last_of('.'));
	size_t tiret = f.find('-');
	if (tiret != string::npos) f = f.substr(tiret + 1);
	return f;
}

// 1. chargement des photos + points AKAZE (fait une seule fois)
void chargerReferences(const string& dossier) {
	vector<String> fichiers, tmp;
	glob(dossier + "/*.png", tmp, false);
	fichiers.insert(fichiers.end(), tmp.begin(), tmp.end());
	glob(dossier + "/*.jpg", tmp, false);
	fichiers.insert(fichiers.end(), tmp.begin(), tmp.end());

	for (int i = 0; i < fichiers.size(); i++) {
		Mat img = lirePhoto(fichiers[i]);
		if (img.empty()) continue;

		vector<KeyPoint> kp;
		Mat desc;
		akaze->detectAndCompute(img, noArray(), kp, desc);

		trainImages.push_back(img);
		noms.push_back(nomDepuisFichier(fichiers[i]));
		kp_train.push_back(kp);
		desc_train.push_back(desc);
		objectCorners.push_back({ Point2f(0, 0), Point2f(img.cols, 0),
			Point2f(img.cols, img.rows), Point2f(0, img.rows) });

		cout << noms.back() << " : " << kp.size() << " points AKAZE" << endl;
	}

	trackingActif.assign(trainImages.size(), false);
	sceneCorners.assign(trainImages.size(), vector<Point2f>());
	ptsRef.assign(trainImages.size(), vector<Point2f>());
	ptsScene.assign(trainImages.size(), vector<Point2f>());
	inliers.assign(trainImages.size(), 0);
}

// 2 detection d'une voiture : matching + homographie
// renvoie le nb d'inliers (0 si pas trouvee)
int akazeTracker(int i, vector<KeyPoint>& kp_query, Mat& desc_query, vector<Point2f>& coins, Mat& H,
	vector<Point2f>& inRef, vector<Point2f>& inScene) {
	if (desc_train[i].empty() || desc_query.rows < 2) return 0;

	// knn (k=2) + ratio test
	vector<vector<DMatch>> matches;
	int64 debutProfil = getTickCount();
	matcher->knnMatch(desc_query, desc_train[i], matches, 2);
	tempsAppariement += msDepuis(debutProfil);

	vector<Point2f> pts_query_good_match, pts_train_good_match;
	for (int j = 0; j < matches.size(); j++) {
		if (matches[j].size() == 2 && matches[j][0].distance < RATIO * matches[j][1].distance) {
			pts_query_good_match.push_back(kp_query[matches[j][0].queryIdx].pt);
			pts_train_good_match.push_back(kp_train[i][matches[j][0].trainIdx].pt);
		}
	}
	//  avec moins de MIN_INLIERS bons appariements, la voiture ne peut pas etre validee : RANSAC inutile
	if (pts_query_good_match.size() < MIN_INLIERS) return 0;

	// homographie avec RANSAC
	Mat masque;
	debutProfil = getTickCount();
	H = findHomography(pts_train_good_match, pts_query_good_match, RANSAC, 3, masque, ITERATIONS_RANSAC);
	tempsRansac += msDepuis(debutProfil);
	appelsRansac++;
	if (H.empty()) return 0;

	// on garde les inliers (ils serviront au tracking)
	inRef.clear();
	inScene.clear();
	for (int j = 0; j < masque.rows; j++) {
		if (masque.at<uchar>(j)) {
			inRef.push_back(pts_train_good_match[j]);
			inScene.push_back(pts_query_good_match[j]);
		}
	}
	int nbInliers = inRef.size();

	// coins de la photo dans l'image camera
	perspectiveTransform(objectCorners[i], coins, H);

	// cadre bizarre (croise ou trop petit) -> on rejette
	if (!isContourConvex(coins) || contourArea(coins) < 1000) return 0;

	return nbInliers;
}

// 3. tracking d'une voiture avec le flot optique
// renvoie false si on l'a perdue
bool suivreVoiture(int i, Mat& grayFrame) {
	vector<Point2f> nextPts, retour;
	vector<uchar> status, status2;
	vector<float> err;

	// flot aller (image d'avant -> maintenant) puis retour
	calcOpticalFlowPyrLK(prevGray, grayFrame, ptsScene[i], nextPts, status, err);
	calcOpticalFlowPyrLK(grayFrame, prevGray, nextPts, retour, status2, err);

	// on garde les points qui reviennent a moins de 3 px
	vector<Point2f> bonsRef, bonsScene;
	for (int k = 0; k < nextPts.size(); k++) {
		if (status[k] && status2[k] && norm(retour[k] - ptsScene[i][k]) < 3) {
			bonsRef.push_back(ptsRef[i][k]);
			bonsScene.push_back(nextPts[k]);
		}
	}
	if (bonsRef.size() < 8) return false;        // plus assez de points

	// nouvelle homographie + nouveaux coins
	Mat masque;
	Mat H = findHomography(bonsRef, bonsScene, RANSAC, 3, masque);
	if (H.empty()) return false;
	vector<Point2f> coins;
	perspectiveTransform(objectCorners[i], coins, H);
	if (!isContourConvex(coins) || contourArea(coins) < 1000) return false;

	// on garde que les inliers pour l'image suivante
	ptsRef[i].clear();
	ptsScene[i].clear();
	for (int k = 0; k < masque.rows; k++) {
		if (masque.at<uchar>(k)) {
			ptsRef[i].push_back(bonsRef[k]);
			ptsScene[i].push_back(bonsScene[k]);
		}
	}
	if (ptsRef[i].size() < 8) return false;

	sceneCorners[i] = coins;
	return true;
}

// 4. fonction principale du module (appelee par la tache 2)
vector<Detection> detecterVoitures(const Mat& frame) {
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	// a) on suit les voitures deja trouvees
	bool aucune = true;
	int64 debutProfil = getTickCount();
	for (int i = 0; i < trainImages.size(); i++) {
		if (trackingActif[i] && !prevGray.empty()) {
			if (!suivreVoiture(i, grayFrame))
				trackingActif[i] = false;     // voiture perdue
		}
		if (trackingActif[i]) aucune = false;
	}
	tempsSuivi += msDepuis(debutProfil);

	// b) detection AKAZE si rien n'est suivi ou toutes les 5 images
	//    (nouvelles voitures + correction de la derive)
	vector<bool> detecteeMaintenant(trainImages.size(), false);
	if (aucune || numImage % PERIODE_DETECTION == 0) {
		vector<KeyPoint> kp_query;
		Mat desc_query;
		debutProfil = getTickCount();
		akaze->detectAndCompute(grayFrame, noArray(), kp_query, desc_query);   // calcule une seule fois
		tempsAkaze += msDepuis(debutProfil);
		pointsScene += (long)kp_query.size();
		imagesDetection++;

		for (int i = 0; i < trainImages.size(); i++) {
			vector<Point2f> coins, inRef, inScene;
			Mat H;
			int n = akazeTracker(i, kp_query, desc_query, coins, H, inRef, inScene);
			if (n >= MIN_INLIERS) {
				sceneCorners[i] = coins;
				ptsRef[i] = inRef;
				ptsScene[i] = inScene;
				inliers[i] = n;
				trackingActif[i] = true;
				detecteeMaintenant[i] = true;
			}
		}
	}

	// c) liste des voitures visibles
	vector<Detection> resultat;
	for (int i = 0; i < trainImages.size(); i++) {
		if (!trackingActif[i]) continue;
		Detection d;
		d.id = i;
		d.nom = noms[i];
		d.coins = sceneCorners[i];
		d.H = getPerspectiveTransform(objectCorners[i], sceneCorners[i]);
		d.tracking = !detecteeMaintenant[i];
		d.nbInliers = inliers[i];
		resultat.push_back(d);
	}

	prevGray = grayFrame.clone();
	numImage++;
	return resultat;
}

// TACHE 2 : association des videos, incrustation multi-objets, mesures et precision 

struct Reference
{
    int id = -1;              
    string marque;            
    string cheminImage;
    string cheminVideo;
    VideoCapture video;
    double cadence = 30.0;
    Mat imageCourante; // reutilisee si la boucle va plus vite que la video
    int64 dernierInstant = 0; // 0 : video pas encore lue
    double reste = 0.0;
};

struct ObjetDetecte
{
    int idReference = -1;
    vector<Point2f> coinsProjetes; // haut gauche, haut droit, bas droit, bas gauche
    bool valide = false;
    bool suivi = false; // flot optique (true) ou AKAZE (false)
    int nombreInliers = -1; 
};

// Durees d'une image (ms).
struct Mesures
{
    double capture = 0.0; // inclut l'attente de la camera
    double detection = 0.0;
    double lectureVideo = 0.0;
    double projection = 0.0;
    double composition = 0.0;
    double affichage = 0.0;
    double boucle = 0.0;

    double incrustation() const { return lectureVideo + projection + composition; }
    double traitement() const { return detection + incrustation(); }
};

double millisecondesDepuis(int64 debut)
{
    return (getTickCount() - debut) * 1000.0 / getTickFrequency();
}

string nomFichier(const string& chemin)
{
    const size_t separateur = chemin.find_last_of("/\\");
    return separateur == string::npos ? chemin : chemin.substr(separateur + 1);
}

int extraireNumero(const string& chemin)
{
    const string nom = nomFichier(chemin);
    return (!nom.empty() && isdigit(static_cast<unsigned char>(nom[0]))) ? stoi(nom) : -1;
}

// "10-Range-Rover.mp4" -> "Range-Rover"
string extraireMarque(const string& chemin)
{
    const string nom = nomFichier(chemin);
    const size_t tiret = nom.find('-');
    const size_t point = nom.find_last_of('.');

    if (tiret == string::npos || point == string::npos || point <= tiret + 1) {
        return "";
    }

    return nom.substr(tiret + 1, point - tiret - 1);
}

// Association image/video par le nom de marque (les numeros 10 et 11 sont inverses entre les deux dossiers).
bool chargerVideos(vector<Reference>& references)
{
    references.clear();

    vector<String> cheminsImages, cheminsVideos;

    try {
        glob(dossierImages + "/*.*", cheminsImages);
        glob(dossierVideos + "/*.mp4", cheminsVideos);
    }
    catch (const cv::Exception&) {
        return false;
    }

    for (const String& cheminImage : cheminsImages) {
        Reference reference;
        reference.id = extraireNumero(cheminImage);
        reference.marque = extraireMarque(cheminImage);
        reference.cheminImage = cheminImage;

        if (reference.id < 0 || reference.marque.empty()) {
            cerr << "Nom d'image ignore : " << cheminImage << endl;
            continue;
        }

        for (const String& cheminVideo : cheminsVideos) {
            if (extraireMarque(cheminVideo) == reference.marque) {
                reference.cheminVideo = cheminVideo;
                break;
            }
        }

        if (reference.cheminVideo.empty() || !reference.video.open(reference.cheminVideo)) {
            cerr << "Video introuvable ou illisible pour : " << reference.marque << endl;
            continue;
        }

        const double cadence = reference.video.get(CAP_PROP_FPS);
        if (cadence > 1.0) reference.cadence = cadence;

        references.push_back(reference);
    }

    sort(references.begin(), references.end(), [](const Reference& a, const Reference& b) { return a.id < b.id; });
    return !references.empty();
}

bool lireImageEnBoucle(VideoCapture& video, Mat& image)
{
    if (video.read(image)) {
        return true;
    }

    video.set(CAP_PROP_POS_FRAMES, 0);
    return video.read(image);
}

// Avance la video selon le temps ecoule 
bool lireImageSynchronisee(Reference& reference, Mat& image)
{
    const double pauseMaximale = 0.5; 
    const int sautMaximal = 15;

    const int64 maintenant = getTickCount();
    double imagesDues = 1.0;

    if (reference.dernierInstant != 0) {
        const double ecoule = (maintenant - reference.dernierInstant) / getTickFrequency();
        if (ecoule < pauseMaximale) imagesDues = reference.reste + ecoule * reference.cadence;
    }

    reference.dernierInstant = maintenant;

    int nombre = static_cast<int>(imagesDues);
    reference.reste = imagesDues - nombre;

    if (nombre == 0 && !reference.imageCourante.empty()) {
        image = reference.imageCourante;
        return true;
    }

    nombre = min(max(nombre, 1), sautMaximal);

    for (int i = 0; i < nombre - 1; ++i) {
        if (!reference.video.grab()) reference.video.set(CAP_PROP_POS_FRAMES, 0);
    }

    if (!lireImageEnBoucle(reference.video, reference.imageCourante)) {
        return false;
    }

    image = reference.imageCourante;
    return true;
}

void afficherReferences(const vector<Reference>& references)
{
    cout << references.size() << " references chargees :" << endl;

    for (const Reference& reference : references) {
        cout << cv::format("  id %2d | %-11s | %-18s | %-18s | %4d x %-4d | %4d images | %.0f img/s",
            reference.id, reference.marque.c_str(), nomFichier(reference.cheminImage).c_str(), nomFichier(reference.cheminVideo).c_str(),
            static_cast<int>(reference.video.get(CAP_PROP_FRAME_WIDTH)), static_cast<int>(reference.video.get(CAP_PROP_FRAME_HEIGHT)),
            static_cast<int>(reference.video.get(CAP_PROP_FRAME_COUNT)), reference.video.get(CAP_PROP_FPS)) << endl;
    }
}

int trouverReference(const vector<Reference>& references, int id)
{
    for (size_t i = 0; i < references.size(); ++i) {
        if (references[i].id == id) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

// Association par le nom de marque : Detection::id n'est pas le numero du fichier.
vector<ObjetDetecte> convertirDetections(const vector<Detection>& detections, const vector<Reference>& references)
{
    vector<ObjetDetecte> objets;

    for (const Detection& detection : detections) {
        for (const Reference& reference : references) {
            if (reference.marque == detection.nom) {
                ObjetDetecte objet;
                objet.idReference = reference.id;
                objet.coinsProjetes = detection.coins;
                objet.valide = true;
                objet.suivi = detection.tracking;
                objet.nombreInliers = detection.nbInliers;
                objets.push_back(objet);
                break;
            }
        }
    }

    return objets;
}

// Repli sur DirectShow si le mode par defaut echoue.
bool ouvrirWebcam(VideoCapture& webcam)
{
    Mat essai;

    if (webcam.open(0) && webcam.read(essai)) {
        return true;
    }

    return webcam.open(0, CAP_DSHOW) && webcam.read(essai);
}

vector<Point2f> coinsImage(const Size& taille)
{
    const float droite = static_cast<float>(taille.width - 1);
    const float bas = static_cast<float>(taille.height - 1);
    return { Point2f(0.0f, 0.0f), Point2f(droite, 0.0f), Point2f(droite, bas), Point2f(0.0f, bas) };
}

// Objets fixes pour tester la tache 2 sans detection.
vector<ObjetDetecte> simulerObjets(const Size& tailleScene, int nombre)
{
    const int ids[3] = { 1, 2, 3 }; // Toyota, BMW, Honda
    const Point2f proportions[3][4] = {
        { { 0.05f, 0.08f }, { 0.45f, 0.12f }, { 0.43f, 0.45f }, { 0.04f, 0.48f } },
        { { 0.55f, 0.12f }, { 0.95f, 0.06f }, { 0.96f, 0.50f }, { 0.57f, 0.44f } },
        { { 0.30f, 0.58f }, { 0.72f, 0.55f }, { 0.75f, 0.95f }, { 0.27f, 0.92f } }
    };

    vector<ObjetDetecte> objets;

    for (int i = 0; i < min(nombre, 3); ++i) {
        ObjetDetecte objet;
        objet.idReference = ids[i];
        objet.valide = true;

        for (const Point2f& proportion : proportions[i]) {
            objet.coinsProjetes.emplace_back(proportion.x * tailleScene.width, proportion.y * tailleScene.height);
        }

        objets.push_back(objet);
    }

    return objets;
}

bool quadrilatereValide(const vector<Point2f>& coins)
{
    if (coins.size() != 4) {
        return false;
    }

    for (const Point2f& coin : coins) {
        if (!isfinite(coin.x) || !isfinite(coin.y)) {
            return false;
        }
    }

    return isContourConvex(coins) && abs(contourArea(coins)) >= 500.0;
}

bool incrusterVideo(const Mat& imageVideo, const vector<Point2f>& coins, Mat& scene, Mesures& mesures)
{
    if (imageVideo.empty() || scene.empty() || imageVideo.type() != scene.type() || !quadrilatereValide(coins)) {
        return false;
    }

    int64 debut = getTickCount();
    const Mat homographie = getPerspectiveTransform(coinsImage(imageVideo.size()), coins);

    // BORDER_REPLICATE : pas de lisere noir au bord du quadrilatere
    Mat videoProjetee;
    warpPerspective(imageVideo, videoProjetee, homographie, scene.size(), INTER_LINEAR, BORDER_REPLICATE);
    mesures.projection += millisecondesDepuis(debut);

    debut = getTickCount();

    vector<Point> contour;
    for (const Point2f& coin : coins) {
        contour.emplace_back(cvRound(coin.x), cvRound(coin.y));
    }

    Mat masque = Mat::zeros(scene.size(), CV_8UC1);
    fillPoly(masque, vector<vector<Point>>{ contour }, Scalar(255));

    // masque geometrique : les pixels noirs de la video sont copies aussi
    videoProjetee.copyTo(scene, masque);
    mesures.composition += millisecondesDepuis(debut);
    return true;
}

void ecrireTexte(Mat& image, const string& texte, const Point& position)
{
    putText(image, texte, position, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 3, LINE_AA);
    putText(image, texte, position, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1, LINE_AA);
}

void dessinerObjet(Mat& scene, const ObjetDetecte& objet, const Reference& reference)
{
    vector<Point> contour;
    for (const Point2f& coin : objet.coinsProjetes) {
        contour.emplace_back(cvRound(coin.x), cvRound(coin.y));
    }

    polylines(scene, contour, true, Scalar(0, 255, 0), 2, LINE_AA);
    for (const Point& sommet : contour) {
        circle(scene, sommet, 4, Scalar(0, 0, 255), FILLED, LINE_AA);
    }

    const string mode = objet.nombreInliers < 0 ? "simule" : objet.suivi ? "Tracking" : cv::format("AKAZE %d", objet.nombreInliers);
    ecrireTexte(scene, cv::format("%d - %s | %s", reference.id, reference.marque.c_str(), mode.c_str()), contour[0] + Point(5, 20));
}

// Renvoie le nombre d'objets augmentes.
int augmenterScene(vector<Reference>& references, const vector<ObjetDetecte>& objets, Mat& scene, Mesures& mesures)
{
    int nombreAugmentes = 0;
    Mat imageVideo;

    for (const ObjetDetecte& objet : objets) {
        const int indice = objet.valide ? trouverReference(references, objet.idReference) : -1;

        if (indice < 0 || !quadrilatereValide(objet.coinsProjetes)) {
            continue;
        }

        Reference& reference = references[indice];

        const int64 debut = getTickCount();
        const bool imageLue = lireImageSynchronisee(reference, imageVideo);
        mesures.lectureVideo += millisecondesDepuis(debut);

        if (!imageLue || !incrusterVideo(imageVideo, objet.coinsProjetes, scene, mesures)) {
            continue;
        }

        dessinerObjet(scene, objet, reference);
        ++nombreAugmentes;
    }

    return nombreAugmentes;
}

double moyenne(const vector<double>& valeurs)
{
    double somme = 0.0;
    for (double valeur : valeurs) somme += valeur;
    return valeurs.empty() ? 0.0 : somme / valeurs.size();
}

double centile(vector<double> valeurs, double proportion)
{
    if (valeurs.empty()) return 0.0;
    sort(valeurs.begin(), valeurs.end());
    return valeurs[static_cast<size_t>(proportion * (valeurs.size() - 1) + 0.5)];
}

// Precision : objets detectes compares aux photos declarees visibles par l'operateur.
struct Evaluation
{
    vector<int> idsAttendus;
    int images = 0;
    int vraisPositifs = 0;      
    int fauxPositifs = 0;       
    int manques = 0;           
};


double precision(const Evaluation& e)
{
    const int total = e.vraisPositifs + e.fauxPositifs;
    return total > 0 ? static_cast<double>(e.vraisPositifs) / total : -1.0;
}

double rappel(const Evaluation& e)
{
    const int total = e.vraisPositifs + e.manques;
    return total > 0 ? static_cast<double>(e.vraisPositifs) / total : -1.0;
}

string pourcentage(double valeur)
{
    return valeur < 0.0 ? "n/d" : cv::format("%.1f %%", 100.0 * valeur);
}

bool contient(const vector<int>& ids, int id)
{
    return find(ids.begin(), ids.end(), id) != ids.end();
}

Evaluation saisirEvaluation(const vector<Reference>& references)
{
    Evaluation evaluation;
    string ligne;

    cout << endl << "Evaluation : numeros des photos visibles, separes par des espaces (ex. 1 9), Entree seule si aucune : " << flush;
    getline(cin, ligne);

    istringstream flux(ligne);
    int id;

    while (flux >> id) {
        if (trouverReference(references, id) >= 0 && !contient(evaluation.idsAttendus, id)) {
            evaluation.idsAttendus.push_back(id);
        }
        else {
            cout << "Numero ignore : " << id << endl;
        }
    }

    cout << "Evaluation lancee avec " << evaluation.idsAttendus.size() << " photo(s) declaree(s)." << endl;
    return evaluation;
}

void compterImage(Evaluation& evaluation, const vector<ObjetDetecte>& objets)
{
    vector<int> idsDetectes;

    for (const ObjetDetecte& objet : objets) {
        idsDetectes.push_back(objet.idReference);
        contient(evaluation.idsAttendus, objet.idReference) ? ++evaluation.vraisPositifs : ++evaluation.fauxPositifs;
    }

    for (int id : evaluation.idsAttendus) {
        if (!contient(idsDetectes, id)) ++evaluation.manques;
    }

    ++evaluation.images;
}

void afficherBilanPrecision(const vector<Evaluation>& evaluations)
{
    cout << endl << "Bilan - precision de la detection (verite terrain declaree par l'operateur)" << endl;
    cout << "sequence | photos declarees | images | vrais positifs | faux positifs | manques | precision | rappel" << endl;

    Evaluation total;

    for (size_t i = 0; i < evaluations.size(); ++i) {
        const Evaluation& e = evaluations[i];
        if (e.images == 0) continue; 

        string declarees;
        for (int id : e.idsAttendus) declarees += cv::format("%d ", id);

        cout << cv::format("%8d | %-16s | %6d | %14d | %13d | %7d | %9s | %6s", static_cast<int>(i) + 1, declarees.empty() ? "aucune" : declarees.c_str(),
            e.images, e.vraisPositifs, e.fauxPositifs, e.manques, pourcentage(precision(e)).c_str(), pourcentage(rappel(e)).c_str()) << endl;

        total.images += e.images;
        total.vraisPositifs += e.vraisPositifs;
        total.fauxPositifs += e.fauxPositifs;
        total.manques += e.manques;
    }

    if (total.images == 0) {
        cout << "aucune sequence d'evaluation complete." << endl;
        return;
    }

    cout << cv::format("   total | %-16s | %6d | %14d | %13d | %7d | %9s | %6s", "", total.images, total.vraisPositifs, total.fauxPositifs, total.manques,
        pourcentage(precision(total)).c_str(), pourcentage(rappel(total)).c_str()) << endl;
    cout << "precision = vrais positifs / (vrais positifs + faux positifs) ; rappel = vrais positifs / (vrais positifs + manques)." << endl;
}

void afficherBilan(const string& titre, const vector<vector<Mesures>>& historique)
{
    cout << endl << "Bilan - " << titre << " (ms par image)" << endl;
    cout << "objets | images | capture | detection | lecture | projection | composition | affichage | traitement moy / med / p95 / max | boucle | FPS reel" << endl;

    for (size_t nombre = 0; nombre < historique.size(); ++nombre) {
        const vector<Mesures>& mesures = historique[nombre];
        if (mesures.empty()) continue;

        vector<double> capture, detection, lecture, projection, composition, affichage, traitement, boucle;

        for (const Mesures& m : mesures) {
            capture.push_back(m.capture);
            detection.push_back(m.detection);
            lecture.push_back(m.lectureVideo);
            projection.push_back(m.projection);
            composition.push_back(m.composition);
            affichage.push_back(m.affichage);
            traitement.push_back(m.traitement());
            boucle.push_back(m.boucle);
        }

        cout << cv::format("%6d | %6d | %7.2f | %9.2f | %7.2f | %10.2f | %11.2f | %9.2f | %14.2f / %5.2f / %5.2f / %5.2f | %6.2f | %8.1f",
            static_cast<int>(nombre), static_cast<int>(mesures.size()), moyenne(capture), moyenne(detection), moyenne(lecture), moyenne(projection),
            moyenne(composition), moyenne(affichage), moyenne(traitement), centile(traitement, 0.5), centile(traitement, 0.95), centile(traitement, 1.0),
            moyenne(boucle), 1000.0 / moyenne(boucle)) << endl;
    }
}

void afficherProfilDetection()
{
    if (imagesDetection == 0) return;

    const double akazeMoyen = tempsAkaze / imagesDetection, appariementMoyen = tempsAppariement / imagesDetection, ransacMoyen = tempsRansac / imagesDetection;
    const double total = akazeMoyen + appariementMoyen + ransacMoyen;

    cout << endl << "Profil de la detection (moyenne par image de detection AKAZE, " << imagesDetection << " images sur " << numImage << ")" << endl;
    cout << cv::format("  extraction AKAZE de la scene      : %7.2f ms (%4.1f %%), %ld points par image", akazeMoyen, 100.0 * akazeMoyen / total, pointsScene / imagesDetection) << endl;
    cout << cv::format("  appariement avec les 12 photos    : %7.2f ms (%4.1f %%)", appariementMoyen, 100.0 * appariementMoyen / total) << endl;
    cout << cv::format("  homographies RANSAC               : %7.2f ms (%4.1f %%), %.1f appels par image sur 12 photos", ransacMoyen, 100.0 * ransacMoyen / total, static_cast<double>(appelsRansac) / imagesDetection) << endl;
    cout << cv::format("  total des trois etapes            : %7.2f ms", total) << endl;
    cout << cv::format("  suivi par flot optique            : %7.2f ms par image (toutes les images)", tempsSuivi / max(1, numImage)) << endl;
    cout << "  coeurs logiques : " << getNumberOfCPUs() << " | fils utilises par OpenCV : " << getNumThreads() << endl;
}

int main(int argc, char** argv)
{
    if (argc >= 3) {
        dossierImages = argv[1];
        dossierVideos = argv[2];
    }

    vector<Reference> references;

    if (!chargerVideos(references)) {
        cerr << "Aucune reference chargee : verifier les dossiers \"" << dossierImages << "\" et \"" << dossierVideos << "\"." << endl;
        return 1;
    }

    afficherReferences(references);

    akaze = AKAZE::create();
    matcher = DescriptorMatcher::create("BruteForce-Hamming");
    chargerReferences(dossierImages);

    if (trainImages.empty()) {
        cerr << "Aucune photo de reference chargee pour la detection : verifier le dossier \"" << dossierImages << "\"." << endl;
        return 1;
    }

    VideoCapture webcam;

    if (!ouvrirWebcam(webcam)) {
        cerr << "Impossible d'ouvrir la webcam ou d'en lire une image." << endl;
        return 1;
    }

    cout << "Touches : d = detection reelle, s = objets simules (0, 1, 2, 3 = leur nombre), e = evaluer la precision, Echap = quitter et afficher les bilans." << endl;

    const string nomFenetre = "VOTAA - Vision augmentee multi-objets";
    const Size tailleScene(640, 480); // taille de travail de la tache 1
    const int imagesEchauffement = 30; // images ignorees apres un changement de mode
    const int imagesEvaluation = 300; // environ 10 s
    const int imagesReprise = 10; // images ignorees apres la saisie
    const int nombreMaximum = static_cast<int>(references.size());
    vector<Evaluation> evaluations;
    bool evaluationEnCours = false;
    int imagesAIgnorer = 0;
    vector<vector<Mesures>> historiqueDetection(nombreMaximum + 1), historiqueSimulation(nombreMaximum + 1);
    bool modeDetection = true;
    int nombreSimules = 1, imagesDepuisChangement = 0;
    int imagesFps = 0;
    double fpsReel = 0.0, traitementPrecedent = 0.0;
    int64 debutFps = getTickCount();
    Mat scene;

    while (true) {
        const int64 debutBoucle = getTickCount();
        Mesures mesures;

        int64 debut = getTickCount();
        const bool sceneLue = webcam.read(scene) && !scene.empty();

        if (!sceneLue) {
            cerr << "Lecture de la webcam impossible." << endl;
            break;
        }

        resize(scene, scene, tailleScene);
        mesures.capture = millisecondesDepuis(debut);

        vector<ObjetDetecte> objets;

        if (modeDetection) {
            debut = getTickCount();
            objets = convertirDetections(detecterVoitures(scene), references); // une seule fois par image
            mesures.detection = millisecondesDepuis(debut);

            if (evaluationEnCours && imagesAIgnorer-- <= 0) {
                compterImage(evaluations.back(), objets);
                evaluationEnCours = evaluations.back().images < imagesEvaluation;
                if (!evaluationEnCours) cout << "Evaluation terminee : precision " << pourcentage(precision(evaluations.back())) << ", rappel " << pourcentage(rappel(evaluations.back())) << endl;
            }
        }
        else {
            objets = simulerObjets(scene.size(), nombreSimules);
        }

        const int nombreAugmentes = augmenterScene(references, objets, scene, mesures);

        debut = getTickCount();
        ecrireTexte(scene, cv::format("%s | objets augmentes : %d / %d", modeDetection ? "Detection" : "Simulation", nombreAugmentes, static_cast<int>(objets.size())), Point(10, scene.rows - 36));
        ecrireTexte(scene, cv::format("Traitement : %.1f ms | FPS reel : %.1f", traitementPrecedent, fpsReel), Point(10, scene.rows - 12));

        if (evaluationEnCours) {
            const Evaluation& e = evaluations.back();
            ecrireTexte(scene, cv::format("Evaluation %d / %d | precision %s | rappel %s", e.images, imagesEvaluation, pourcentage(precision(e)).c_str(), pourcentage(rappel(e)).c_str()), Point(10, scene.rows - 60));
        }
        imshow(nomFenetre, scene);
        const int touche = waitKey(1);
        mesures.affichage = millisecondesDepuis(debut);
        mesures.boucle = millisecondesDepuis(debutBoucle);

        traitementPrecedent = mesures.traitement();

        if (++imagesDepuisChangement > imagesEchauffement) {
            (modeDetection ? historiqueDetection : historiqueSimulation)[nombreAugmentes].push_back(mesures);
        }

        ++imagesFps;
        if (millisecondesDepuis(debutFps) >= 1000.0) {
            fpsReel = imagesFps * 1000.0 / millisecondesDepuis(debutFps);
            imagesFps = 0;
            debutFps = getTickCount();
        }

        if (touche == 27) break;

        if (touche == 'e') {
            evaluations.push_back(saisirEvaluation(references));
            evaluationEnCours = true;
            imagesAIgnorer = imagesReprise;
            imagesDepuisChangement = 0;
            imagesFps = 0;
            debutFps = getTickCount();
        }

        const bool nouveauMode = (touche == 'd' || touche == 'e') ? true : (touche == 's' || (touche >= '0' && touche <= '3')) ? false : modeDetection;
        const int nouveauNombre = (touche >= '0' && touche <= '3') ? touche - '0' : nombreSimules;

        if (nouveauMode != modeDetection || nouveauNombre != nombreSimules) {
            modeDetection = nouveauMode;
            nombreSimules = nouveauNombre;
            if (!modeDetection) evaluationEnCours = false;
            imagesDepuisChangement = 0;
        }
    }

    destroyAllWindows();
    afficherBilan("detection reelle ", historiqueDetection);
    afficherBilan("objets simules ", historiqueSimulation);
    cout << "traitement = detection + lecture + projection + composition ; FPS reel = images / duree totale des boucles (capture et affichage inclus)." << endl;
    afficherProfilDetection();
    afficherBilanPrecision(evaluations);
    return 0;
}
