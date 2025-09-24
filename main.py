from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import base64
from pdf2image import convert_from_bytes
from typing import Dict, List
import logging
from pydantic import BaseModel
import subprocess
import asyncio
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stamp and Sign Detection API",
    description="API для обнаружения печатей и подписей на изображениях, PDF и Word документах",
    version="1.1.0"  # Обновили версию
)

# Загрузка модели
try:
    model = YOLO("best.pt")
    logger.info("✅ Модель успешно загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    raise


class BinaryFileRequest(BaseModel):
    filename: str
    data: str


class DocumentProcessor:
    def __init__(self, model):
        self.model = model
        self.class_names = {15: "stamp", 16: "sign"}

    async def process_file(self, file_bytes: bytes, filename: str) -> Dict:
        """Обработка файла (изображение, PDF или Word)"""
        try:
            file_ext = os.path.splitext(filename.lower())[1]

            if file_ext in ['.docx', '.doc']:
                raise HTTPException(
                    status_code=400,
                    detail="Формат Word не поддерживается. Используйте изображения или PDF."
                )
            elif file_ext == '.pdf':
                images = await self._convert_pdf_to_images(file_bytes)
                results = []
                for img in images:
                    result = await self._process_image(img)
                    results.append(result)
                return self._aggregate_results(results)
            else:
                image = await self._bytes_to_image(file_bytes)
                return await self._process_image(image)

        except Exception as e:
            logger.error(f"Ошибка обработки файла: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

    async def _convert_pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Конвертация PDF в изображения"""
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=200,
                fmt='jpeg',
                thread_count=4
            )
            return images
        except Exception as e:
            logger.error(f"Ошибка конвертации PDF: {e}")
            raise HTTPException(status_code=400, detail="Невозможно конвертировать PDF")

    async def _bytes_to_image(self, file_bytes: bytes) -> Image.Image:
        """Конвертация bytes в PIL Image"""
        try:
            return Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            logger.error(f"Ошибка чтения изображения: {e}")
            raise HTTPException(status_code=400, detail="Невозможно прочитать изображение")

    async def _process_image(self, image: Image.Image) -> Dict:
        """Обработка одного изображения"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results = self.model(img_array, conf=0.2)
            return self._analyze_detections(results[0])

        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            raise HTTPException(status_code=500, detail="Ошибка обработки изображения")

    def _analyze_detections(self, result) -> Dict:
        """Анализ результатов детекции"""
        detections = {
            "stamp": 0,
            "sign": 0,
            "total_objects": 0,
            "details": []
        }

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()

                class_name = self.class_names.get(class_id, "unknown")

                detections[class_name] += 1
                detections["total_objects"] += 1

                detections["details"].append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [round(coord, 2) for coord in bbox]
                })

        return detections

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Агрегация результатов по всем страницам"""
        total_stamps = sum(result["stamp"] for result in results)
        total_signs = sum(result["sign"] for result in results)
        total_objects = sum(result["total_objects"] for result in results)

        return {
            "total_stamps": total_stamps,
            "total_signs": total_signs,
            "total_objects": total_objects,
            "pages_processed": len(results),
            "per_page_results": results
        }


# Инициализация процессора
processor = DocumentProcessor(model)


@app.get("/")
async def root():
    return {"message": "Stamp and Sign Detection API", "status": "active", "version": "1.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True, "word_support": True}


@app.get("/supported-formats")
async def supported_formats():
    """Возвращает поддерживаемые форматы"""
    return {
        "supported_formats": {
            "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "documents": [".pdf"]
        }
    }


@app.post("/detect-url")
async def detect_objects(
        file: UploadFile = File(..., description="Загрузите изображение, PDF или Word файл")
):
    """
    Обнаружение печатей и подписей на изображении, PDF или Word документе
    """
    try:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
        file_ext = os.path.splitext(file.filename.lower())[1]

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
            )

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")

        result = await processor.process_file(file_bytes, file.filename)
        logger.info(f"✅ Обработан файл {file.filename}")

        return JSONResponse(content={
            "filename": file.filename,
            "success": True,
            "results": result
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@app.post("/detect-binary")
async def detect_objects_binary(
        file_request: BinaryFileRequest
):
    """
    Обнаружение печатей и подписей из бинарных данных
    """
    try:
        # РАЗРЕШАЕМ Word в binary тоже!
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
        file_ext = os.path.splitext(file_request.filename.lower())[1]

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
            )

        try:
            file_bytes = base64.b64decode(file_request.data)
        except Exception:
            raise HTTPException(status_code=400, detail="Некорректные base64 данные")

        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Пустой файл")

        result = await processor.process_file(file_bytes, file_request.filename)
        logger.info(f"✅ Обработан бинарный файл {file_request.filename}")

        return {
            "filename": file_request.filename,
            "success": True,
            "results": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке бинарных данных: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

