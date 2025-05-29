# Test Data Directory

This directory contains test files for the NuroPadel testing suite.

## Structure

```
data/
├── videos/          # Test video files
├── images/          # Test image files
├── expected/        # Expected output files for validation
└── samples/         # Sample data for unit tests
```

## Test Videos

For API testing, we use publicly available sample videos:
- Small video files (< 5MB) for quick tests
- Various resolutions and formats
- Short duration videos (< 30 seconds)

## Sample URLs for Testing

```python
# Quick test videos
SAMPLE_VIDEOS = [
    "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
    "https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4",
]
```

## Adding Test Data

1. Keep test files small (< 10MB each)
2. Use common formats (MP4, JPG, PNG)
3. Include files with different characteristics:
   - Different resolutions
   - Different frame rates
   - Different lighting conditions
   - Different number of people/objects

## Usage in Tests

```python
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent.parent / "data"
test_video = TEST_DATA_DIR / "videos" / "sample.mp4"