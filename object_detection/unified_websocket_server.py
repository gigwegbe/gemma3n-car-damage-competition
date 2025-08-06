#!/usr/bin/env python3
"""
Unified WebSocket Server for YOLO Detection + Gemma3n Analysis
Combines both APIs into a single WebSocket-based service
"""

import asyncio
import websockets
import json
import base64
import uuid
import os
import tempfile
from datetime import datetime
import requests
from pathlib import Path
import logging

# YOLO imports
from ultralytics import YOLO
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedServer:
    def __init__(self):
        # YOLO Configuration
        self.yolo_model = None
        self.load_yolo_model()
        
        # Directory setup
        self.temp_dir = Path("temp_uploads")
        self.output_dir = Path("outputs")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # YOLO class configuration
        self.CLASS_NAMES = [
            "dent", "scratch", "crack",
            "glass shatter", "lamp broken", "tire flat"
        ]
        
        self.CLASS_WEIGHTS = {
            "dent": 2.0,
            "scratch": 2.0,
            "crack": 3.0,
            "glass shatter": 5.0,
            "lamp broken": 3.5,
            "tire flat": 1.0
        }
        
        # Gemma3n API configuration
        self.gemma_api_url = "http://localhost:6000/analyze"
        
    def load_yolo_model(self):
        """Load YOLO model"""
        try:
            model_path = "yolov11/train6/weights/best.pt"
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                logger.info("✅ YOLO model loaded successfully!")
            else:
                logger.error(f"❌ YOLO model not found at: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error loading YOLO model: {e}")
    
    def process_detections(self, results):
        """Process YOLO results into sorted detections with severity scores"""
        detections = []
        
        if len(results) == 0 or results[0].boxes is None:
            return detections
            
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self.CLASS_NAMES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = width * height

            severity_score = self.CLASS_WEIGHTS[cls_name] * conf * area

            detections.append({
                "class": cls_name,
                "conf": conf,
                "bbox": (x1.item(), y1.item(), x2.item(), y2.item()),
                "area": area,
                "severity_score": severity_score
            })

        # Sort by severity
        sorted_detections = sorted(detections, key=lambda x: x["severity_score"], reverse=True)
        return sorted_detections
    
    def format_detections_string(self, sorted_detections):
        """Format detections as a string with the exact formatting style"""
        if not sorted_detections:
            return "No detections found."
        
        formatted_lines = ["Ranked Detections by Severity:"]
        
        for det in sorted_detections:
            line = (f"{det['class']} | Conf: {det['conf']:.2f} | Area: {det['area']:.0f} | "
                    f"Score: {det['severity_score']:.0f} | BBox: {det['bbox']}")
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    async def run_yolo_detection(self, image_data, websocket, request_id):
        """Run YOLO detection on the image"""
        try:
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Running YOLO detection...",
                "request_id": request_id,
                "step": "yolo_detection",
                "timestamp": datetime.now().isoformat()
            }))
            
            if self.yolo_model is None:
                raise Exception("YOLO model not loaded")
            
            # Save image temporarily
            unique_id = str(uuid.uuid4())
            temp_path = self.temp_dir / f"{unique_id}.jpg"
            
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            # Run YOLO inference
            results = self.yolo_model.predict(
                source=str(temp_path),
                imgsz=1024,
                conf=0.20,
                iou=0.50,
                save=True,
                project=str(self.output_dir),
                name=unique_id,
                exist_ok=True
            )
            
            # Process detections
            sorted_detections = self.process_detections(results)
            formatted_detections = self.format_detections_string(sorted_detections)
            
            # Find output image path
            output_image_path = None
            output_dir = self.output_dir / unique_id
            if output_dir.exists():
                for file in output_dir.iterdir():
                    if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        output_image_path = os.path.abspath(str(file))  # Use absolute path
                        break
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            # Convert output image to base64 if available
            output_image_base64 = None
            if output_image_path and os.path.exists(output_image_path):
                with open(output_image_path, "rb") as img_file:
                    output_image_base64 = base64.b64encode(img_file.read()).decode()
            
            return {
                "detections": sorted_detections,
                "formatted_detections": formatted_detections,
                "detection_count": len(sorted_detections),
                "output_image_path": output_image_path,
                "output_image_base64": output_image_base64
            }
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"YOLO detection failed: {str(e)}",
                "request_id": request_id
            }))
            return None
    
    async def run_gemma_analysis(self, output_image_path, formatted_detections, websocket, request_id):
        """Send results to Gemma3n API for analysis"""
        try:
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Running Gemma3n analysis...",
                "request_id": request_id,
                "step": "gemma_analysis",
                "timestamp": datetime.now().isoformat()
            }))
            
            if not output_image_path or not os.path.exists(output_image_path):
                raise Exception("Output image not available for analysis")
            
            logger.info(f"📸 Using absolute image path for Gemma analysis: {output_image_path}")
            
            # Prepare files and data for Gemma API
            with open(output_image_path, 'rb') as img_file:
                files = {'file': img_file}
                data = {'question': formatted_detections}
                
                # Make request to Gemma API
                response = requests.post(
                    self.gemma_api_url,
                    files=files,
                    data=data,
                    timeout=60  # 60 second timeout
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('answer', 'No analysis returned')
            else:
                raise Exception(f"Gemma API returned status {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Gemma3n API. Make sure it's running on port 6000"
            logger.error(error_msg)
            await websocket.send(json.dumps({
                "type": "error",
                "message": error_msg,
                "request_id": request_id
            }))
            return None
        except Exception as e:
            error_msg = f"Gemma analysis failed: {str(e)}"
            logger.error(error_msg)
            await websocket.send(json.dumps({
                "type": "error",
                "message": error_msg,
                "request_id": request_id
            }))
            return None
    
    async def run_gemma_analysis_and_send_results(self, yolo_results, websocket, request_id):
        """Run Gemma analysis and send results to client"""
        try:
            gemma_analysis = await self.run_gemma_analysis(
                yolo_results["output_image_path"],
                yolo_results["formatted_detections"],
                websocket,
                request_id
            )
            
            if gemma_analysis:
                # Send final complete results
                await websocket.send(json.dumps({
                    "type": "complete_analysis",
                    "data": {
                        "yolo_results": {
                            "detections": yolo_results["detections"],
                            "formatted_detections": yolo_results["formatted_detections"],
                            "detection_count": yolo_results["detection_count"],
                            "output_image": yolo_results["output_image_base64"]
                        },
                        "gemma_analysis": gemma_analysis
                    },
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }))
            else:
                # Send notification that Gemma analysis failed
                await websocket.send(json.dumps({
                    "type": "gemma_analysis_failed",
                    "message": "YOLO detection completed successfully, but Gemma analysis failed. Check if Gemma API is running on port 6000.",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(f"Error in Gemma analysis task: {e}")
            await websocket.send(json.dumps({
                "type": "gemma_analysis_failed",
                "message": f"Gemma analysis failed: {str(e)}",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }))
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        client_id = str(uuid.uuid4())[:8]
        remote_address = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"🔌 Client {client_id} connected from {remote_address}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connection",
                "message": "Connected to Unified YOLO + Gemma3n Server",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            async for message in websocket:
                try:
                    # Log message size for debugging large uploads
                    message_size_mb = len(message) / (1024 * 1024)
                    if message_size_mb > 1:  # Log if message > 1MB
                        logger.info(f"📨 Large message from client {client_id}: {message_size_mb:.1f}MB")
                    
                    data = json.loads(message)
                    request_id = data.get('request_id', str(uuid.uuid4()))
                    
                    if data.get('type') == 'analyze_image':
                        await self.process_full_analysis(websocket, data, request_id)
                    elif data.get('type') == 'ping':
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Unknown message type: {data.get('type')}",
                            "request_id": request_id
                        }))
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error from client {client_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format - please check your message"
                    }))
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Server error processing request: {str(e)}"
                    }))
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔌 Client {client_id} disconnected: {e.code} {e.reason}")
        except websockets.exceptions.ConnectionClosedError:
            logger.info(f"🔌 Client {client_id} connection closed unexpectedly")
        except Exception as e:
            logger.error(f"Client {client_id} unexpected error: {e}")
        finally:
            logger.info(f"🔌 Cleaning up resources for client {client_id}")
    
    async def process_full_analysis(self, websocket, data, request_id):
        """Process the complete workflow: YOLO → Gemma3n → Results"""
        try:
            # Step 1: Extract image data
            image_data = data.get('image')
            if not image_data:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "No image data provided",
                    "request_id": request_id
                }))
                return
            
            # Step 2: Run YOLO detection
            yolo_results = await self.run_yolo_detection(image_data, websocket, request_id)
            if not yolo_results:
                return  # Error already sent
            
            # Send YOLO results to client IMMEDIATELY
            await websocket.send(json.dumps({
                "type": "yolo_results",
                "data": {
                    "detections": yolo_results["detections"],
                    "formatted_detections": yolo_results["formatted_detections"],
                    "detection_count": yolo_results["detection_count"],
                    "output_image": yolo_results["output_image_base64"]
                },
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Step 3: Run Gemma3n analysis in parallel if we have detections
            if yolo_results["detection_count"] > 0:
                # Start Gemma analysis immediately (non-blocking)
                asyncio.create_task(
                    self.run_gemma_analysis_and_send_results(
                        yolo_results, websocket, request_id
                    )
                )
            else:
                # No detections found - send final result
                await websocket.send(json.dumps({
                    "type": "no_detections",
                    "data": {
                        "detections": [],
                        "formatted_detections": "No detections found.",
                        "detection_count": 0,
                        "output_image": yolo_results["output_image_base64"]
                    },
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }))
            
        except Exception as e:
            logger.error(f"Full analysis error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Analysis failed: {str(e)}",
                "request_id": request_id
            }))

async def main():
    """Main server function"""
    server = UnifiedServer()
    
    # Create a handler function that wraps the server method
    async def websocket_handler(websocket):
        await server.handle_client(websocket)
    
    print("🚀 Starting Unified WebSocket Server...")
    print("📡 Server Configuration:")
    print(f"   - WebSocket Port: 8765")
    print(f"   - YOLO Model: {'✅ Loaded' if server.yolo_model else '❌ Not loaded'}")
    print(f"   - Gemma3n API: {server.gemma_api_url}")
    print(f"   - Temp Directory: {server.temp_dir}")
    print(f"   - Output Directory: {server.output_dir}")
    print("\n🔧 Workflow:")
    print("   1. Client sends image via WebSocket")
    print("   2. YOLO detection runs")
    print("   3. Results sent to Gemma3n API")
    print("   4. Complete analysis returned to client")
    print("\n📋 WebSocket Messages:")
    print("   - Send: {type: 'analyze_image', image: 'base64_data', request_id: 'optional'}")
    print("   - Receive: Multiple status updates and final results")
    print(f"\n🌐 WebSocket URL: ws://0.0.0.0:8765 (Public: ws://206.168.81.71:8765)")
    
    # Configure WebSocket server with larger message limits and timeout settings
    server_config = {
        # "host": "0.0.0.0",  # Bind to all interfaces for public access
        "port": 8765,
        "max_size": 50 * 1024 * 1024,  # 50MB max message size
        "max_queue": 100,  # Increased message queue
        # "read_limit": 2**20,  # 1MB read buffer
        "write_limit": 2**20,  # 1MB write buffer
        "ping_interval": 30,  # 30 second ping interval
        "ping_timeout": 10,  # 10 second ping timeout
        "close_timeout": 10,  # 10 second close timeout
    }
    
    async with websockets.serve(websocket_handler, **server_config):
        print("✅ Server is running with enhanced configuration:")
        print(f"   - Max message size: {server_config['max_size'] // (1024*1024)}MB")
        print(f"   - Ping interval: {server_config['ping_interval']}s")
        print("   - Ready for large image uploads!")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
