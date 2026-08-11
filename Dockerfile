FROM python:3.12-slim

WORKDIR /code

# Install backend dependencies first so this layer is cached across rebuilds
# that only change application code, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what app/main.py actually needs at runtime -- no notebooks, no
# frontend/, no raw dataset CSV.
COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
