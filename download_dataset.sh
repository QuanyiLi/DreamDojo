#!/bin/bash
# Download GR1_robot dataset in batches to avoid HuggingFace rate limits
# Each batch downloads one chunk (data + videos), then waits 5 minutes

set -e
source .venv/bin/activate

DEST="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1"
REPO="nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1"

# Total chunks: 000 to ~021 (based on 44k files / ~1000 per chunk)
for i in $(seq -w 0 21); do
    CHUNK="chunk-0${i}"
    echo "$(date): Downloading $CHUNK..."

    # Download data parquet files for this chunk
    huggingface-cli download "$REPO" \
        --include "GR1_robot/data/${CHUNK}/**" \
        --repo-type dataset \
        --local-dir "$DEST" 2>&1 || {
        echo "Rate limited on data ${CHUNK}, waiting 6 minutes..."
        sleep 360
        huggingface-cli download "$REPO" \
            --include "GR1_robot/data/${CHUNK}/**" \
            --repo-type dataset \
            --local-dir "$DEST" 2>&1 || echo "Failed data ${CHUNK}, continuing..."
    }

    # Download video files for this chunk
    huggingface-cli download "$REPO" \
        --include "GR1_robot/videos/${CHUNK}/**" \
        --repo-type dataset \
        --local-dir "$DEST" 2>&1 || {
        echo "Rate limited on videos ${CHUNK}, waiting 6 minutes..."
        sleep 360
        huggingface-cli download "$REPO" \
            --include "GR1_robot/videos/${CHUNK}/**" \
            --repo-type dataset \
            --local-dir "$DEST" 2>&1 || echo "Failed videos ${CHUNK}, continuing..."
    }

    echo "$(date): Finished $CHUNK"

    # Wait between chunks to stay under rate limit
    echo "Waiting 5 minutes before next chunk..."
    sleep 300
done

echo "$(date): Download complete!"
