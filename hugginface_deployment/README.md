Running the VLM Deployment Script (MacBook(`mps`) or RTX3090(`cuda`)):

Follow the steps below to deploy the AutoVision Inspector VLM based on your device:

-  Install Device-Specific Requirements: 

    For MacBook:
    ```
    pip install -r requirements_mac.txt
    ```

    For RTX3090:
    ```
    pip install -r requirements_rtx.txt
    ```
- Run the Deployment Script
  
    For MacBook:
    ```
    python3 hf_run_macbook.py 
    ```

    For RTX3090:
    ```
    python3 hf_run_rtx.py 
    ```