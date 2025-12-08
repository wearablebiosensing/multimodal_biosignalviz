# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# FIX 1: Globally disable pip's progress bar to prevent build crashes
ENV PIP_PROGRESS_BAR=off

# FIX 2: Disable Streamlit's file watcher to prevent runtime crashes
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# FIX 3: Restrict numerical libraries (numpy/scipy) to single-threaded mode
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- COPY CONFIGURATION KEYS ---
# Copy Firebase Key (Database)
COPY firebase_key.json .
# Copy Gemini Key (AI Agent) - CRITICAL FIX
COPY gemini_key.txt .

# --- COPY APPLICATION CODE ---
# Copy the main visualizer script
COPY biosignal_viz.py .
# Copy the agent logic script - CRITICAL FIX
COPY agent.py .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "biosignal_viz.py", "--server.port=8501", "--server.address=0.0.0.0"]