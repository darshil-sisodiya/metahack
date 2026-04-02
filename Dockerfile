FROM python:3.11-slim

# Create the non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy the files and give ownership to the new user
COPY --chown=user:user . .

RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user before running the app
USER user

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
