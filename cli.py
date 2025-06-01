import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image, ImageDraw
import moviepy.editor as mp
from moviepy.video.fx import resize
import os
import mediapipe as mp_face
from scipy.signal import savgol_filter
import math


class SmartVideoCropper:
    def __init__(self):
        """
        Système avancé de détection de visages avec recentrage intelligent et auto-zoom
        """
        # Modèle Hugging Face pour détection d'objets
        self.object_detector = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=0 if torch.cuda.is_available() else -1,
        )

        # MediaPipe pour détection précise des visages
        self.mp_face_detection = mp_face.solutions.face_detection
        self.mp_drawing = mp_face.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Configuration du format de sortie (9:16 pour shorts)
        self.target_aspect_ratio = 9 / 16
        self.output_width = 1080
        self.output_height = 1920

        # Paramètres de zoom adaptatif
        self.zoom_levels = {
            "wide": 1.0,  # Vue large
            "medium": 1.3,  # Vue moyenne
            "close": 1.6,  # Vue rapprochée
            "extreme": 2.0,  # Vue très rapprochée
        }

        # Historique pour le lissage
        self.tracking_history = []
        self.zoom_history = []

    def detect_faces_advanced(self, frame):
        """
        Détection avancée combinant DETR et MediaPipe
        """
        faces = []

        # 1. Détection avec MediaPipe (plus précise pour les visages)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Convertir les coordonnées relatives en absolues
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                faces.append(
                    {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "confidence": detection.score[0],
                        "type": "face",
                        "area": width * height,
                    }
                )

        # 2. Détection avec DETR pour les personnes (backup et contexte)
        pil_image = Image.fromarray(rgb_frame)
        detections = self.object_detector(pil_image)

        for detection in detections:
            if detection["label"] == "person" and detection["score"] > 0.6:
                box = detection["box"]
                width = box["xmax"] - box["xmin"]
                height = box["ymax"] - box["ymin"]

                faces.append(
                    {
                        "x": box["xmin"],
                        "y": box["ymin"],
                        "width": width,
                        "height": height,
                        "confidence": detection["score"],
                        "type": "person",
                        "area": width * height,
                    }
                )

        return faces

    def calculate_smart_crop_center(self, faces, frame_width, frame_height, frame_idx):
        """
        Calcule le centre intelligent avec priorisation des visages
        """
        if not faces:
            return frame_width // 2, frame_height // 2, "wide"

        # Séparer les visages des personnes
        face_detections = [f for f in faces if f["type"] == "face"]
        person_detections = [f for f in faces if f["type"] == "person"]

        # Prioriser les visages détectés
        primary_targets = face_detections if face_detections else person_detections

        if not primary_targets:
            return frame_width // 2, frame_height // 2, "wide"

        # Calculer le centre pondéré
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        max_area = 0

        for target in primary_targets:
            # Pondération basée sur la confiance et la taille
            weight = target["confidence"] * (
                1 + target["area"] / (frame_width * frame_height)
            )
            center_x = target["x"] + target["width"] // 2
            center_y = target["y"] + target["height"] // 2

            weighted_x += center_x * weight
            weighted_y += center_y * weight
            total_weight += weight
            max_area = max(max_area, target["area"])

        center_x = (
            int(weighted_x / total_weight) if total_weight > 0 else frame_width // 2
        )
        center_y = (
            int(weighted_y / total_weight) if total_weight > 0 else frame_height // 2
        )

        # Déterminer le niveau de zoom basé sur la taille et le nombre de visages
        zoom_level = self.determine_zoom_level(
            primary_targets, frame_width, frame_height
        )

        return center_x, center_y, zoom_level

    def determine_zoom_level(self, targets, frame_width, frame_height):
        """
        Détermine le niveau de zoom optimal selon le contexte
        """
        if not targets:
            return "wide"

        # Calculer les métriques
        num_faces = len([t for t in targets if t["type"] == "face"])
        avg_face_area = (
            np.mean([t["area"] for t in targets if t["type"] == "face"])
            if num_faces > 0
            else 0
        )
        total_area = sum(t["area"] for t in targets)
        frame_area = frame_width * frame_height

        # Logique de zoom adaptatif
        if num_faces == 0:
            return "wide"
        elif num_faces == 1:
            # Un seul visage : zoom selon sa taille
            face_ratio = avg_face_area / frame_area
            if face_ratio > 0.15:
                return "close"
            elif face_ratio > 0.08:
                return "medium"
            else:
                return "wide"
        elif num_faces == 2:
            # Deux visages : zoom moyen
            return "medium"
        else:
            # Plusieurs visages : vue large
            return "wide"

    def smooth_tracking_advanced(self, centers, zoom_levels, window_size=7):
        """
        Lissage avancé avec filtrage Savitzky-Golay
        """
        if len(centers) < window_size:
            return centers, zoom_levels

        # Extraire les coordonnées
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]

        # Appliquer le filtre Savitzky-Golay
        try:
            smoothed_x = savgol_filter(x_coords, window_size, 3)
            smoothed_y = savgol_filter(y_coords, window_size, 3)
        except:
            # Fallback sur moyenne mobile
            smoothed_x = self.moving_average(x_coords, window_size)
            smoothed_y = self.moving_average(y_coords, window_size)

        smoothed_centers = [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]

        # Lisser les transitions de zoom
        smoothed_zoom = self.smooth_zoom_transitions(zoom_levels)

        return smoothed_centers, smoothed_zoom

    def smooth_zoom_transitions(self, zoom_levels):
        """
        Lisse les transitions de zoom pour éviter les changements brusques
        """
        if len(zoom_levels) < 3:
            return zoom_levels

        smoothed = []
        for i, current_zoom in enumerate(zoom_levels):
            if i == 0 or i == len(zoom_levels) - 1:
                smoothed.append(current_zoom)
                continue

            # Vérifier la cohérence avec les frames voisines
            prev_zoom = zoom_levels[i - 1]
            next_zoom = zoom_levels[i + 1] if i + 1 < len(zoom_levels) else current_zoom

            # Si le zoom actuel est différent des voisins, utiliser le précédent
            if current_zoom != prev_zoom and current_zoom != next_zoom:
                smoothed.append(prev_zoom)
            else:
                smoothed.append(current_zoom)

        return smoothed

    def moving_average(self, data, window_size):
        """
        Moyenne mobile simple
        """
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            avg = sum(data[start_idx:end_idx]) / (end_idx - start_idx)
            result.append(avg)
        return result

    def apply_smart_crop(self, frame, center_x, center_y, zoom_level):
        """
        Applique le recadrage intelligent avec zoom adaptatif
        """
        height, width = frame.shape[:2]
        zoom_factor = self.zoom_levels[zoom_level]

        # Calculer les dimensions effectives après zoom
        effective_width = int(width / zoom_factor)
        effective_height = int(height / zoom_factor)

        # Maintenir le ratio d'aspect cible
        if effective_width / effective_height > self.target_aspect_ratio:
            # Trop large, ajuster la largeur
            crop_width = int(effective_height * self.target_aspect_ratio)
            crop_height = effective_height
        else:
            # Trop haut, ajuster la hauteur
            crop_width = effective_width
            crop_height = int(effective_width / self.target_aspect_ratio)

        # Calculer les coordonnées de recadrage
        left = max(0, center_x - crop_width // 2)
        right = min(width, left + crop_width)
        left = max(0, right - crop_width)

        top = max(0, center_y - crop_height // 2)
        bottom = min(height, top + crop_height)
        top = max(0, bottom - crop_height)

        # Recadrer
        cropped = frame[top:bottom, left:right]

        # Appliquer le zoom si nécessaire
        if zoom_factor > 1.0:
            # Calculer les dimensions finales après zoom
            final_width = int(crop_width * zoom_factor)
            final_height = int(crop_height * zoom_factor)

            # Redimensionner puis recadrer au centre
            resized = cv2.resize(cropped, (final_width, final_height))

            # Recadrer au centre pour obtenir les dimensions finales
            start_x = (final_width - crop_width) // 2
            start_y = (final_height - crop_height) // 2
            cropped = resized[
                start_y : start_y + crop_height, start_x : start_x + crop_width
            ]

        # Redimensionner au format de sortie final
        final_frame = cv2.resize(cropped, (self.output_width, self.output_height))

        return final_frame

    def add_zoom_indicators(self, frame, zoom_level, center_x, center_y):
        """
        Ajoute des indicateurs visuels pour le debug (optionnel)
        """
        # Couleurs pour chaque niveau de zoom
        colors = {
            "wide": (0, 255, 0),  # Vert
            "medium": (255, 255, 0),  # Jaune
            "close": (255, 165, 0),  # Orange
            "extreme": (255, 0, 0),  # Rouge
        }

        color = colors.get(zoom_level, (255, 255, 255))

        # Dessiner un cercle au centre
        cv2.circle(frame, (center_x, center_y), 10, color, -1)

        # Ajouter le texte du niveau de zoom
        cv2.putText(
            frame,
            f"Zoom: {zoom_level}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        return frame

    def process_video_smart(
        self, input_path, output_path, sample_rate=3, debug_mode=False
    ):
        """
        Traitement intelligent de la vidéo avec toutes les fonctionnalités avancées
        """
        print(f"🎬 Traitement intelligent de: {input_path}")

        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (self.output_width, self.output_height)
        )

        centers = []
        zoom_levels = []
        frame_count = 0

        print("🔍 Phase 1: Analyse intelligente des visages...")

        # Première passe: analyse complète
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Détection avancée
                faces = self.detect_faces_advanced(frame)
                center_x, center_y, zoom_level = self.calculate_smart_crop_center(
                    faces, frame.shape[1], frame.shape[0], frame_count
                )

                centers.append((center_x, center_y))
                zoom_levels.append(zoom_level)

                if frame_count % (sample_rate * 15) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"   📊 Progression: {progress:.1f}% - Visages détectés: {len([f for f in faces if f['type'] == 'face'])}"
                    )

            frame_count += 1

        print("🎯 Phase 2: Optimisation du tracking et du zoom...")

        # Interpoler pour toutes les frames
        all_centers = []
        all_zoom_levels = []
        center_idx = 0

        for i in range(total_frames):
            if i % sample_rate == 0 and center_idx < len(centers):
                all_centers.append(centers[center_idx])
                all_zoom_levels.append(zoom_levels[center_idx])
                center_idx += 1
            else:
                if len(all_centers) > 0:
                    all_centers.append(all_centers[-1])
                    all_zoom_levels.append(all_zoom_levels[-1])
                else:
                    all_centers.append((frame.shape[1] // 2, frame.shape[0] // 2))
                    all_zoom_levels.append("wide")

        # Lissage avancé
        smoothed_centers, smoothed_zoom = self.smooth_tracking_advanced(
            all_centers, all_zoom_levels
        )

        print("🎥 Phase 3: Génération de la vidéo avec auto-zoom...")

        # Deuxième passe: application du recadrage intelligent
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count < len(smoothed_centers):
                center_x, center_y = smoothed_centers[frame_count]
                zoom_level = smoothed_zoom[frame_count]

                # Appliquer le recadrage intelligent
                smart_cropped = self.apply_smart_crop(
                    frame, center_x, center_y, zoom_level
                )

                # Ajouter des indicateurs de debug si demandé
                if debug_mode:
                    smart_cropped = self.add_zoom_indicators(
                        smart_cropped, zoom_level, self.output_width // 2, 50
                    )

                out.write(smart_cropped)

            if frame_count % 60 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   🎬 Rendu: {progress:.1f}%")

            frame_count += 1

        cap.release()
        out.release()

        print(f"✅ Vidéo intelligente sauvegardée: {output_path}")

        # Ajouter l'audio
        self.add_audio_advanced(input_path, output_path)

        # Statistiques finales
        self.print_processing_stats(smoothed_zoom)

    def add_audio_advanced(self, original_path, cropped_path):
        """
        Ajoute l'audio avec gestion avancée des erreurs
        """
        try:
            print("🔊 Ajout de l'audio...")
            original_clip = mp.VideoFileClip(original_path)
            cropped_clip = mp.VideoFileClip(cropped_path)

            if original_clip.audio is not None:
                final_clip = cropped_clip.set_audio(original_clip.audio)
                temp_path = cropped_path.replace(".mp4", "_with_audio.mp4")
                final_clip.write_videofile(temp_path, verbose=False, logger=None)

                os.replace(temp_path, cropped_path)
                print("✅ Audio ajouté avec succès!")
            else:
                print("ℹ️  Aucun audio détecté dans la vidéo source")

            original_clip.close()
            cropped_clip.close()

        except Exception as e:
            print(f"⚠️  Erreur audio (vidéo fonctionnelle): {e}")

    def print_processing_stats(self, zoom_levels):
        """
        Affiche les statistiques de traitement
        """
        from collections import Counter

        zoom_stats = Counter(zoom_levels)
        total_frames = len(zoom_levels)

        print("\n📈 Statistiques de traitement:")
        print("=" * 40)
        for zoom_level, count in zoom_stats.items():
            percentage = (count / total_frames) * 100
            print(
                f"   {zoom_level.capitalize():>8}: {percentage:5.1f}% ({count} frames)"
            )
        print("=" * 40)


def main():
    """
    Interface principale avec options avancées
    """
    print("🚀 Système de Recadrage Vidéo Intelligent")
    print("=" * 50)

    cropper = SmartVideoCropper()

    # Configuration
    input_video = "input_video.mp4"
    output_video = "smart_cropped_video.mp4"

    # Options avancées
    sample_rate = 2  # Plus précis mais plus lent
    debug_mode = False  # Afficher les indicateurs de zoom

    if os.path.exists(input_video):
        print(f"📁 Fichier trouvé: {input_video}")
        cropper.process_video_smart(input_video, output_video, sample_rate, debug_mode)
        print("\n🎉 Traitement terminé avec succès!")
        print(f"📹 Vidéo de sortie: {output_video}")
    else:
        print(f"❌ Fichier introuvable: {input_video}")
        print("📝 Instructions:")
        print("   1. Placez votre vidéo dans le dossier du script")
        print("   2. Renommez-la 'input_video.mp4'")
        print("   3. Relancez le script")


if __name__ == "__main__":
    print("📦 Dépendances requises:")
    print(
        "   pip install transformers torch opencv-python pillow moviepy mediapipe scipy"
    )
    print()

    main()
