# Image de base avec Python 3.9
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    # Dépendances OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dépendances pour le traitement vidéo
    ffmpeg \
    # Nettoyage
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Mise à jour pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Créer le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY cli.py .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Volume pour les vidéos
VOLUME ["/app/videos"]

# Point d'entrée
ENTRYPOINT ["python", "cli.py"]