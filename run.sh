#!/bin/bash

# Créer le dossier videos s'il n'existe pas
mkdir -p videos

# Construire l'image Docker
echo "🔨 Construction de l'image Docker..."
docker build -t smart-video-cropper .

# Exécuter le conteneur
echo "🚀 Lancement du conteneur..."
docker run --rm -v "$(pwd)/videos:/app/videos" smart-video-cropper
