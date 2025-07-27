# Enhanced screenshot capabilities
"""
Screen Capture utilities for Android devices
Handles screenshot capture and image processing
"""

import time
import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .adb_manager import ADBManager

class ScreenCapture:
    """Enhanced screen capture with Pixel 5 optimizations"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.adb = ADBManager()
        self.pixel5_config = {
            'width': 1080,
            'height': 2340,
            'density': 440
        }
    
    def capture_screenshot(self) -> Optional[Image.Image]:
        """Capture screenshot and return as PIL Image"""
        
        try:
            screenshot_bytes = self.adb.get_screenshot(self.device_id)
            
            if screenshot_bytes:
                image = Image.open(io.BytesIO(screenshot_bytes))
                print(f"📸 Screenshot captured: {image.size}")
                return image
            else:
                print("⚠️ No screenshot data received")
                return None
                
        except Exception as e:
            print(f"❌ Screenshot capture error: {e}")
            return None
    
    def capture_with_annotations(self, ui_elements: list = None) -> Optional[Image.Image]:
        """Capture screenshot with UI element annotations"""
        
        image = self.capture_screenshot()
        if not image:
            return None
        
        if ui_elements:
            return self._annotate_image(image, ui_elements)
        
        return image
    
    def _annotate_image(self, image: Image.Image, ui_elements: list) -> Image.Image:
        """Annotate image with UI element bounds"""
        
        try:
            # Create a copy for annotation
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw bounds for each element
            for i, element in enumerate(ui_elements[:20]):  # Limit to first 20 elements
                bounds = element.get('bounds', [])
                if len(bounds) >= 2:
                    x1, y1 = bounds[0]
                    x2, y2 = bounds[1]
                    
                    # Choose color based on element type
                    color = self._get_element_color(element)
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Add label if element has text
                    text = element.get('text', '') or element.get('content_desc', '')
                    if text and len(text) < 20:
                        draw.text((x1, y1-15), text[:15], fill=color, font=font)
                    
                    # Add number label
                    draw.text((x1+2, y1+2), str(i+1), fill=color, font=font)
            
            return annotated
            
        except Exception as e:
            print(f"❌ Image annotation error: {e}")
            return image
    
    def _get_element_color(self, element: Dict[str, Any]) -> str:
        """Get color for element based on its properties"""
        
        if element.get('clickable', False):
            return 'red'
        elif element.get('text', ''):
            return 'blue'
        elif 'Button' in element.get('class', ''):
            return 'green'
        elif 'EditText' in element.get('class', ''):
            return 'orange'
        else:
            return 'gray'
    
    def capture_element_screenshot(self, element_bounds: list) -> Optional[Image.Image]:
        """Capture screenshot of specific UI element"""
        
        full_image = self.capture_screenshot()
        if not full_image or len(element_bounds) < 2:
            return None
        
        try:
            x1, y1 = element_bounds[0]
            x2, y2 = element_bounds[1]
            
            # Crop to element bounds
            cropped = full_image.crop((x1, y1, x2, y2))
            return cropped
            
        except Exception as e:
            print(f"❌ Element screenshot error: {e}")
            return None
    
    def compare_screenshots(self, image1: Image.Image, image2: Image.Image) -> Dict[str, Any]:
        """Compare two screenshots and return difference metrics"""
        
        try:
            # Ensure images are same size
            if image1.size != image2.size:
                image2 = image2.resize(image1.size)
            
            # Convert to numpy arrays
            arr1 = np.array(image1)
            arr2 = np.array(image2)
            
            # Calculate differences
            diff = np.abs(arr1.astype(float) - arr2.astype(float))
            
            # Calculate metrics
            metrics = {
                'mean_difference': float(np.mean(diff)),
                'max_difference': float(np.max(diff)),
                'similarity_percentage': 100.0 - (np.mean(diff) / 255.0 * 100.0),
                'changed_pixels': int(np.sum(diff > 10)),  # Threshold for significant change
                'total_pixels': arr1.size
            }
            
            metrics['change_percentage'] = (metrics['changed_pixels'] / metrics['total_pixels']) * 100.0
            
            return metrics
            
        except Exception as e:
            print(f"❌ Screenshot comparison error: {e}")
            return {}
    
    def detect_ui_changes(self, before_image: Image.Image, after_image: Image.Image, 
                         threshold: float = 5.0) -> bool:
        """Detect if UI has changed significantly between screenshots"""
        
        comparison = self.compare_screenshots(before_image, after_image)
        change_percentage = comparison.get('change_percentage', 0.0)
        
        return change_percentage > threshold
    
    def save_screenshot_sequence(self, count: int = 5, interval: float = 1.0, 
                               prefix: str = "screen") -> list:
        """Capture sequence of screenshots"""
        
        screenshots = []
        
        for i in range(count):
            image = self.capture_screenshot()
            if image:
                filename = f"{prefix}_{i+1:03d}_{int(time.time())}.png"
                image.save(filename)
                screenshots.append(filename)
                print(f"📸 Saved screenshot {i+1}/{count}: {filename}")
            
            if i < count - 1:
                time.sleep(interval)
        
        return screenshots
    
    def capture_scrolling_screenshot(self, scroll_count: int = 3) -> Optional[Image.Image]:
        """Capture screenshot while scrolling to get full content"""
        
        screenshots = []
        
        try:
            # Capture initial screenshot
            initial = self.capture_screenshot()
            if not initial:
                return None
            
            screenshots.append(initial)
            
            # Perform scrolling and capture
            for i in range(scroll_count):
                # Scroll down
                self.adb.swipe(
                    self.device_id,
                    self.pixel5_config['width'] // 2,
                    self.pixel5_config['height'] * 3 // 4,
                    self.pixel5_config['width'] // 2,
                    self.pixel5_config['height'] // 4
                )
                
                time.sleep(1)
                
                # Capture screenshot
                screenshot = self.capture_screenshot()
                if screenshot:
                    screenshots.append(screenshot)
            
            # Combine screenshots vertically
            if len(screenshots) > 1:
                return self._combine_screenshots_vertically(screenshots)
            else:
                return screenshots[0] if screenshots else None
                
        except Exception as e:
            print(f"❌ Scrolling screenshot error: {e}")
            return screenshots[0] if screenshots else None
    
    def _combine_screenshots_vertically(self, screenshots: list) -> Image.Image:
        """Combine multiple screenshots into one vertical image"""
        
        if not screenshots:
            return None
        
        # Calculate total height
        total_height = sum(img.height for img in screenshots)
        max_width = max(img.width for img in screenshots)
        
        # Create combined image
        combined = Image.new('RGB', (max_width, total_height))
        
        y_offset = 0
        for img in screenshots:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
        
        return combined
    
    def extract_text_regions(self, image: Image.Image = None) -> list:
        """Extract potential text regions from screenshot"""
        
        if not image:
            image = self.capture_screenshot()
        
        if not image:
            return []
        
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(gray)
            
            # Simple text region detection using edge detection
            # This is a basic implementation - could be enhanced with OCR
            from PIL import ImageFilter
            
            # Apply edge detection
            edges = image.filter(ImageFilter.FIND_EDGES)
            
            # Convert to array and find regions with high edge density
            edge_array = np.array(edges.convert('L'))
            
            # Find rectangular regions with high edge density
            # This is a simplified approach
            text_regions = []
            
            # Divide image into grid and check edge density
            grid_size = 50
            height, width = edge_array.shape
            
            for y in range(0, height - grid_size, grid_size):
                for x in range(0, width - grid_size, grid_size):
                    region = edge_array[y:y+grid_size, x:x+grid_size]
                    edge_density = np.mean(region)
                    
                    if edge_density > 20:  # Threshold for text-like regions
                        text_regions.append({
                            'bounds': [[x, y], [x + grid_size, y + grid_size]],
                            'edge_density': float(edge_density)
                        })
            
            return text_regions
            
        except Exception as e:
            print(f"❌ Text region extraction error: {e}")
            return []
    
    def get_screen_brightness(self, image: Image.Image = None) -> float:
        """Calculate average screen brightness"""
        
        if not image:
            image = self.capture_screenshot()
        
        if not image:
            return 0.0
        
        try:
            # Convert to grayscale and calculate mean
            gray = image.convert('L')
            brightness = np.mean(np.array(gray)) / 255.0
            
            return float(brightness)
            
        except Exception as e:
            print(f"❌ Brightness calculation error: {e}")
            return 0.0
    
    def detect_loading_indicators(self, image: Image.Image = None) -> list:
        """Detect loading indicators in screenshot"""
        
        if not image:
            image = self.capture_screenshot()
        
        if not image:
            return []
        
        # Simple detection based on circular regions and common loading patterns
        # This is a basic implementation that could be enhanced
        
        indicators = []
        
        try:
            # Convert to HSV for better color detection
            hsv = image.convert('HSV')
            hsv_array = np.array(hsv)
            
            # Look for spinning/rotating patterns (simplified)
            # In a real implementation, this would use more sophisticated detection
            
            # For now, just return empty list
            # Could be enhanced with template matching or ML-based detection
            
        except Exception as e:
            print(f"❌ Loading indicator detection error: {e}")
        
        return indicators
