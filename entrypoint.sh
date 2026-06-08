#!/bin/bash
set -e

echo "Starting backend container..."

# Download model if not exists
if [ ! -f "/app/best_model.pt" ]; then
  echo "Downloading best_model.pt from Supabase..."
  curl -L -H "apikey: $SUPABASE_SERVICE_KEY" \
       "$SUPABASE_URL/storage/v1/object/public/models/best_model.pt" \
       -o /app/best_model.pt
fi

echo "Model is ready! Starting server..."

uvicorn app:app --host 0.0.0.0 --port 8000
