FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
ENV PIP_PROGRESS_BAR=off
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY firebase_key.json .
COPY biosignal_viz.py .
COPY agent.py .
COPY firebase_module.py .
COPY firestore_export.py .
EXPOSE 8501
CMD ["streamlit", "run", "biosignal_viz.py", "--server.port=8501", "--server.address=0.0.0.0"]