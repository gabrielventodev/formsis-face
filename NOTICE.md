# Modelos y archivos de terceros

| Archivo | Origen | Licencia |
|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | YuNet, [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `models/face_recognition_sface_2021dec.onnx` | SFace, [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache 2.0 |
| `models/2.7_80x80_MiniFASNetV2.onnx`, `models/4_0_0_80x80_MiniFASNetV1SE.onnx` | Pesos de [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Minivision), exportados a ONNX con `tools/export_minifasnet.py` | Apache 2.0 |
| `tests/fixtures/*.jpg` | Imágenes de ejemplo de Silent-Face-Anti-Spoofing (`images/sample`) | Apache 2.0 |

Las sumas SHA-256 de los modelos están en `models/SHA256SUMS`; la imagen de Docker no se construye si no coinciden.
