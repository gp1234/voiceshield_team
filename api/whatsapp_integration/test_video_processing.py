#!/usr/bin/env python3
"""
Test script for video processing functionality
Tests video detection and audio extraction
"""
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from video_processor import detect_media_type, extract_audio_from_video, get_media_info, cleanup_temp_file
except ImportError as e:
    print(f"❌ Error: Could not import video_processor module: {e}")
    print("   Make sure you're running this from the whatsapp_integration directory")
    sys.exit(1)


def test_media_type_detection():
    """Test media type detection functionality"""
    print("=== Testing Media Type Detection ===")

    test_cases = [
        ("video/mp4", None, "video"),
        ("video/3gpp", None, "video"),
        ("audio/ogg", None, "audio"),
        ("audio/mpeg", None, "audio"),
        ("application/octet-stream", "test.mp4", "video"),
        ("application/octet-stream", "test.ogg", "audio"),
        ("unknown/type", None, "unknown"),
    ]

    for content_type, filename, expected in test_cases:
        result = detect_media_type(content_type, filename)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(
            f"{status} - {content_type} + {filename} → {result} (expected: {expected})")

    print()


def test_video_audio_extraction():
    """Test video audio extraction with a sample file"""
    print("=== Testing Video Audio Extraction ===")

    # This would require an actual video file to test
    # For now, we'll just test the function exists and handles errors

    print("📝 Note: To test video extraction, place a sample video file in this directory")
    print("📝 and update this function with the correct path")

    # Test with non-existent file
    result = extract_audio_from_video("non_existent_file.mp4")
    if result is None:
        print("✅ PASS - Correctly handles non-existent file")
    else:
        print("❌ FAIL - Should return None for non-existent file")

    print()


def test_cleanup_functionality():
    """Test file cleanup functionality"""
    print("=== Testing Cleanup Functionality ===")

    # Test cleanup with non-existent file
    result = cleanup_temp_file("non_existent_file.tmp")
    if result == False:
        print("✅ PASS - Correctly handles non-existent file cleanup")
    else:
        print("❌ FAIL - Should return False for non-existent file")

    print()


def main():
    """Run all tests"""
    print("🧪 VoiceShield Video Processing Tests")
    print("=" * 50)

    test_media_type_detection()
    test_video_audio_extraction()
    test_cleanup_functionality()

    print("✅ All tests completed!")
    print("\n📋 To fully test video processing:")
    print("   1. Place a sample video file (e.g., sample.mp4) in this directory")
    print("   2. Update test_video_audio_extraction() with the correct path")
    print("   3. Run the test again")


if __name__ == "__main__":
    main()
