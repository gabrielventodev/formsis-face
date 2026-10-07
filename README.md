# formsis-face

Servicio de verificación facial de Formsis. Por ahora hace **prueba de vida** (liveness): decide si quien está frente a la cámara es una persona real y no una foto impresa, una pantalla o un video. Más adelante sumará la comparación de la selfie con la foto del documento.

- Corre 100% en tu servidor, en CPU, sin llamar a ningún servicio externo.
- Es **sin estado**: recibe fotogramas, responde números y no guarda nada. Las fotos y el resultado los guarda el backend (formsis-backend).
- Solo lo llama el backend, por la red interna de Docker. No se publica a internet.

## Cómo decide

El backend elige un **desafío al azar**, por ejemplo `["center", "left", "closer"]`. El portal muestra cada instrucción unos segundos y toma 3 fotogramas mientras está en pantalla. El servicio revisa:

1. **Calidad de cada fotograma**: exactamente una cara, de tamaño suficiente, con buena luz y nítida. Los fotogramas que no cumplen se descartan.
2. **Desafío**: cada paso tiene al menos un fotograma que lo cumple.
   - `center`: mirando de frente. El mejor fotograma de este paso es la **referencia**.
   - `left` / `right`: la cabeza girada hacia **su** izquierda o derecha respecto de la referencia.
   - `closer`: la cara al menos un 18% más ancha que en la referencia.
3. **Misma persona**: todos los fotogramas muestran la misma cara que la referencia (SFace).
4. **Anti-spoofing pasivo**: un modelo (MiniFASNet) mira la textura de la cara en los fotogramas de frente para distinguir piel real de papel o pantalla.

| Decisión | Significado | Qué ve la persona |
|---|---|---|
| `pass` | Persona real, todo en orden. | Verificación completa. |
| `review` | Completó el desafío pero el anti-spoofing quedó en zona gris. Un revisor debe mirar los fotogramas. | Verificación completa. |
| `retry` | No se pudo verificar (sin cara, poca luz, no siguió una instrucción). Nada sospechoso. | Intenta de nuevo, con el motivo. |
| `fail` | Parece una foto o pantalla, o cambió la persona a mitad del desafío. | Intenta de nuevo (el backend limita los intentos). |

Los umbrales están en `app/liveness.py` (`Thresholds`). Son valores iniciales razonables, **no calibrados con datos reales**: hay que ajustarlos midiendo intentos reales y ataques propios.

## API

`GET /healthz` → `{"status":"ok","version":"0.1.0","models":{...}}`

`POST /v1/liveness` (multipart, `Authorization: Bearer $FACE_TOKEN`)

| Campo | Contenido |
|---|---|
| `steps` | JSON, 2 a 5 pasos, empieza con `center`. Ej: `["center","left","closer"]` |
| `frame_steps` | JSON, el índice del paso de cada fotograma, en orden. Ej: `[0,0,0,1,1,1,2,2,2]` |
| `frames` | Los fotogramas JPEG o PNG (repetido), 1 a 20, máx. 5 por paso y 1,5 MB cada uno, todos del mismo tamaño, lado entre 240 y 1920 px |

Respuesta:

```json
{
  "decision": "pass",
  "reasons": [],
  "scores": {"passive": 0.97, "consistency": 0.82},
  "steps": [{"index": 0, "step": "center", "ok": true, "frame": 1}],
  "best_frame": 1,
  "frames": [{"index": 0, "step": 0, "faces": 1, "face_ratio": 0.31, "yaw": 0.04,
              "brightness": 121.0, "sharpness": 85.2, "real": 0.98, "similarity": 1.0, "issues": []}],
  "engine": {"detector": "yunet-2023mar", "antispoof": "minifasnet-v2-2.7+v1se-4.0",
             "recognizer": "sface-2021dec", "service": "0.1.0"},
  "elapsed_ms": 240
}
```

Motivos posibles en `reasons`: `no_face`, `multiple_faces`, `face_too_small`, `too_dark`, `too_bright`, `blurry`, `challenge_not_completed`, `spoof_suspected`, `passive_uncertain`, `face_changed`.

## Ejecutar

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
FACE_TOKEN=dev uvicorn app.main:create_app --factory --port 8000
```

Con Docker (lo levanta el `docker-compose.yml` de formsis-v2 como servicio `face`):

```bash
docker build -t formsis-face .
docker run --rm -e FACE_TOKEN=dev -p 8000:8000 formsis-face
```

| Variable | Uso |
|---|---|
| `FACE_TOKEN` | Obligatoria. El backend la manda como `Authorization: Bearer`. |
| `FACE_ALLOW_NO_TOKEN` | `true` solo para desarrollo local sin token. |
| `FACE_MODELS_DIR` | Carpeta de modelos (por defecto `./models`). |

## Modelos

Están en `models/` con sus sumas en `models/SHA256SUMS`; el Dockerfile se niega a construir si no coinciden. Licencias y origen en [NOTICE.md](NOTICE.md). Los modelos anti-spoofing se exportaron a ONNX con `tools/export_minifasnet.py` y dan el mismo resultado que los pesos originales en PyTorch.

## Límites conocidos

- **No está certificado** (ISO/IEC 30107-3). Frena fotos impresas, pantallas y videos que no siguen el desafío, pero no máscaras 3D bien hechas.
- **Inyección de video** (cámaras virtuales, deepfakes en tiempo real): la norma no la cubre y este servicio tampoco. Se mitiga con el desafío al azar y el tiempo límite que pone el backend.
- Los 5 puntos de la cara de YuNet son ruidosos; por eso los giros se miden contra la referencia del mismo intento y no en valor absoluto.
- Los umbrales necesitan calibración con datos reales (ver el plan `plan/kyc-facial.md` del proyecto).
