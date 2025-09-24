# Document Stamp & Signature Detection API
A high-performance computer vision API for automated document analysis. This service detects stamps and signatures in various document formats using a fine-tuned YOLO model.

## Project Overview
This project addresses the limitations of standard YOLO models in document analysis by:

* Fine-tuning on 1,100 real-world documents
* Adding signature detection capabilities to the existing model
* Improving stamp detection accuracy significantly
* Providing a production-ready API for seamless integration

## Key Features
* Multi-format Support: Process images (JPG, PNG, BMP, TIFF) and PDF documents
* High Accuracy: Enhanced detection through custom training on real documents
* Detailed Analytics: Bounding box coordinates, confidence scores, and object counts
* Scalable Architecture: FastAPI-based async processing
* RESTful API: Easy integration with existing systems

## Technical Stack
* Framework: FastAPI
* Computer Vision: YOLO (Ultralytics)
* Image Processing: OpenCV, PIL
* PDF Support: pdf2image
* Async Processing: asyncio, uvicorn
