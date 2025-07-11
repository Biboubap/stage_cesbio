#!/bin/bash

source logiciels/OTB-9.1.1-Linux/otbenv.profile


# Définir le répertoire parent
BASE_DIR="Konstantin/UAV_Konstantin_Tabatha"

# Pour chaque dossier dans le répertoire de base
for DIR in "$BASE_DIR"/*; do
    if [ -d "$DIR" ]; then
        # Extraire le nom du dossier
        FOLDER_NAME=$(basename "$DIR")
        
        # Ignorer le dossier Chesnay
        if [ "$FOLDER_NAME" == "Chesnay" ]; then
            echo "Dossier Chesnay ignoré - traitement déjà effectué"
            continue
        fi
        
        echo "Traitement du dossier: $FOLDER_NAME"
        
        # Trouver les fichiers
        ORTHO_FILE=$(find "$DIR" -name "*ortho_export*_32615.tif" | head -n 1)
        DSM_FILE=$(find "$DIR" -name "*ElevationToolbox*_32615.tif" | head -n 1)
        
        # Définir le fichier de sortie
        OUTPUT_FILE="$DIR/${FOLDER_NAME}_DSM_Resampled.tif"
        
        # Vérifier que les deux fichiers existent
        if [ -n "$ORTHO_FILE" ] && [ -n "$DSM_FILE" ]; then
            echo "Ortho (référence): $ORTHO_FILE"
            echo "DSM (à transformer): $DSM_FILE"
            echo "Sortie: $OUTPUT_FILE"
            
            # Exécuter otbcli_Superimpose avec les variables d'environnement nécessaires
            echo "Exécution de otbcli_Superimpose..."
            bash otbcli_Superimpose -inr "$ORTHO_FILE" -inm "$DSM_FILE" -out "$OUTPUT_FILE"

            # Vérifier si la commande a réussi
            if [ $? -eq 0 ]; then
                echo "Traitement terminé avec succès pour $FOLDER_NAME"
            else
                echo "ERREUR: Le traitement a échoué pour $FOLDER_NAME"
            fi
            echo "-------------------------"
        else
            echo "ERREUR: Fichiers requis introuvables dans $FOLDER_NAME"
            [ -z "$ORTHO_FILE" ] && echo "Fichier ortho_export manquant"
            [ -z "$DSM_FILE" ] && echo "Fichier ElevationToolbox manquant"
            echo "-------------------------"
        fi
    fi
done

echo "Traitement de tous les dossiers terminé"



#!/bin/bash

# Définir le répertoire parent
BASE_DIR="Konstantin/UAV_Konstantin_Tabatha"

# Pour chaque dossier dans le répertoire de base
for DIR in "$BASE_DIR"/*; do
    if [ -d "$DIR" ]; then
        # Extraire le nom du dossier
        FOLDER_NAME=$(basename "$DIR")
        
        
        echo "Traitement du dossier: $FOLDER_NAME"
        
        # Trouver les fichiers
        ORTHO_FILE=$(find "$DIR" -name "*ortho_export*_32615.tif" | head -n 1)
        DSM_FILE=$(find "$DIR" -name "*RadiometricThermal*_32615.tif" | head -n 1)
        
        # Définir le fichier de sortie
        OUTPUT_FILE="$DIR/${FOLDER_NAME}_Thermal_Resampled.tif"
        
        # Vérifier que les deux fichiers existent
        if [ -n "$ORTHO_FILE" ] && [ -n "$DSM_FILE" ]; then
            echo "Ortho (référence): $ORTHO_FILE"
            echo "DSM (à transformer): $DSM_FILE"
            echo "Sortie: $OUTPUT_FILE"
            
            # Exécuter otbcli_Superimpose avec les variables d'environnement nécessaires
            echo "Exécution de otbcli_Superimpose..."
            bash otbcli_Superimpose -inr "$ORTHO_FILE" -inm "$DSM_FILE" -out "$OUTPUT_FILE"

            # Vérifier si la commande a réussi
            if [ $? -eq 0 ]; then
                echo "Traitement terminé avec succès pour $FOLDER_NAME"
            else
                echo "ERREUR: Le traitement a échoué pour $FOLDER_NAME"
            fi
            echo "-------------------------"
        else
            echo "ERREUR: Fichiers requis introuvables dans $FOLDER_NAME"
            [ -z "$ORTHO_FILE" ] && echo "Fichier ortho_export manquant"
            [ -z "$DSM_FILE" ] && echo "Fichier ElevationToolbox manquant"
            echo "-------------------------"
        fi
    fi
done

echo "Traitement de tous les dossiers terminé"