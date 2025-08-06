### Connecting the AutoVision Inspector VLM to Frontend: 


To connect the `car_damage_analyzer.html` frontend with the AutoVision VLM (hosted in hugginface_deployment), follow these steps:


- Start the WebSocket Server

Run the WebSocket server to bridge the frontend with the AutoVision VLM:
```
python3 unified_websocket_server.py
```
- Serve the Frontend Web App

Start a simple HTTP server to serve the frontend interface:
```
python -m http.server 8000

```

Then, open your browser and navigate to http://localhost:8000/car_damage_analyzer.html to use the application.