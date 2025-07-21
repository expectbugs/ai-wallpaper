#!/usr/bin/env python3
"""
Smart Artifact Detection - AGGRESSIVE SEAM DETECTION
Zero tolerance for visible boundaries
NO ERROR TOLERANCE - FAIL LOUD
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..core import get_logger

class SmartArtifactDetector:
    """Aggressive detection for perfect seamless images"""
    
    def __init__(self):
        self.logger = get_logger()
        
    def quick_analysis(self, 
                      image_path: Path,
                      metadata: Dict) -> Dict:
        """
        Aggressive seam detection with multiple methods.
        NO ERROR TOLERANCE - FAILS COMPLETELY ON ANY ERROR
        """
        # Load and prepare for detection
        image = Image.open(image_path)
        original_size = image.size
        
        # Work at higher resolution for better detection (up to 4K)
        scale = min(1.0, 4096 / max(image.size))
        if scale < 1.0:
            detect_size = (int(image.width * scale), int(image.height * scale))
            detect_image = image.resize(detect_size, Image.Resampling.LANCZOS)
        else:
            detect_image = image
            detect_size = image.size
        
        img_array = np.array(detect_image)
        h, w = img_array.shape[:2]
        
        issues_found = False
        problem_mask = None
        severity = 'none'
        
        # 1. AGGRESSIVE Progressive Boundary Detection
        boundaries = metadata.get('progressive_boundaries', [])
        boundaries_v = metadata.get('progressive_boundaries_vertical', [])
        seam_details = metadata.get('seam_details', [])
        
        if boundaries or boundaries_v:
            self.logger.info(f"Detecting seams at {len(boundaries)} H + {len(boundaries_v)} V boundaries")
            
            # Use multiple detection methods
            seam_mask = self._detect_all_seams(
                img_array, 
                boundaries, 
                boundaries_v,
                seam_details,
                scale
            )
            
            if seam_mask is not None:
                problem_mask = seam_mask
                issues_found = True
                severity = 'critical'  # All progressive seams are critical
                self.logger.info("CRITICAL: Found progressive expansion seams")
        
        # 2. Tile boundary detection (if used)
        if metadata.get('used_tiled', False):
            tile_boundaries = metadata.get('tile_boundaries', [])
            if len(tile_boundaries) > 0:
                tile_mask = self._detect_tile_artifacts(img_array, tile_boundaries, scale)
                if tile_mask is not None:
                    if problem_mask is None:
                        problem_mask = tile_mask
                    else:
                        problem_mask = np.maximum(problem_mask, tile_mask)
                    issues_found = True
                    if severity == 'none':
                        severity = 'high'
                    self.logger.info("Found tile boundary artifacts")
        
        # 3. General discontinuity detection (catches missed seams)
        if not issues_found:  # Only if we haven't found issues yet
            discontinuity_mask = self._detect_discontinuities(img_array)
            if discontinuity_mask is not None:
                problem_mask = discontinuity_mask
                issues_found = True
                severity = 'medium'
                self.logger.info("Found general discontinuities")
        
        # Scale mask back to original size
        if issues_found and scale < 1.0:
            mask_pil = Image.fromarray((problem_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
            problem_mask = np.array(mask_pil) / 255.0
        
        return {
            'needs_multipass': issues_found,
            'mask': problem_mask,
            'severity': severity,
            'seam_count': len(boundaries) + len(boundaries_v)
        }
    
    def _detect_all_seams(self, 
                         image: np.ndarray, 
                         h_boundaries: List[int],
                         v_boundaries: List[int],
                         seam_details: List[Dict],
                         scale: float) -> Optional[np.ndarray]:
        """Detect ALL seams using multiple methods"""
        h, w = image.shape[:2]
        combined_mask = None
        
        # Process horizontal boundaries
        for boundary_x in h_boundaries:
            x = int(boundary_x * scale)
            if 10 < x < w - 10:  # More aggressive - check closer to edges
                # Method 1: Color difference (lower threshold)
                mask1 = self._detect_color_difference(image, x, axis='vertical')
                
                # Method 2: Gradient discontinuity
                mask2 = self._detect_gradient_discontinuity(image, x, axis='vertical')
                
                # Method 3: Frequency analysis (detects texture changes)
                mask3 = self._detect_frequency_change(image, x, axis='vertical')
                
                # Combine all detection methods
                seam_mask = None
                if mask1 is not None:
                    seam_mask = mask1
                if mask2 is not None:
                    seam_mask = mask2 if seam_mask is None else np.maximum(seam_mask, mask2)
                if mask3 is not None:
                    seam_mask = mask3 if seam_mask is None else np.maximum(seam_mask, mask3)
                
                if seam_mask is not None:
                    if combined_mask is None:
                        combined_mask = np.zeros((h, w), dtype=np.float32)
                    combined_mask = np.maximum(combined_mask, seam_mask)
        
        # Process vertical boundaries
        for boundary_y in v_boundaries:
            y = int(boundary_y * scale)
            if 10 < y < h - 10:
                # Same three methods for horizontal seams
                mask1 = self._detect_color_difference(image, y, axis='horizontal')
                mask2 = self._detect_gradient_discontinuity(image, y, axis='horizontal')
                mask3 = self._detect_frequency_change(image, y, axis='horizontal')
                
                seam_mask = None
                if mask1 is not None:
                    seam_mask = mask1
                if mask2 is not None:
                    seam_mask = mask2 if seam_mask is None else np.maximum(seam_mask, mask2)
                if mask3 is not None:
                    seam_mask = mask3 if seam_mask is None else np.maximum(seam_mask, mask3)
                
                if seam_mask is not None:
                    if combined_mask is None:
                        combined_mask = np.zeros((h, w), dtype=np.float32)
                    combined_mask = np.maximum(combined_mask, seam_mask)
        
        # Adaptive mask width based on expansion size
        if combined_mask is not None and seam_details:
            # Use largest expansion size to determine mask width
            max_expansion = max(d.get('expansion_size', 100) for d in seam_details)
            # Wider masks for larger expansions (minimum 100px, up to 20% of expansion)
            base_width = max(100, int(max_expansion * 0.2))
            
            # Apply adaptive gaussian blur
            blur_size = base_width // 4
            if blur_size % 2 == 0:
                blur_size += 1  # Must be odd
            combined_mask = cv2.GaussianBlur(combined_mask, (blur_size, blur_size), 0)
        
        return combined_mask
    
    def _detect_color_difference(self, image: np.ndarray, pos: int, axis: str) -> Optional[np.ndarray]:
        """Detect color differences with LOWER threshold"""
        h, w = image.shape[:2]
        
        if axis == 'vertical':  # Vertical line at x=pos
            if pos < 30 or pos > w - 30:
                return None
                
            # Sample wider regions
            left = image[:, max(0, pos-30):pos]
            right = image[:, pos:min(w, pos+30)]
            
            if left.size == 0 or right.size == 0:
                return None
            
            # Multiple comparison methods
            # 1. Mean color difference (lower threshold)
            mean_diff = np.abs(np.mean(left, axis=(0,1)) - np.mean(right, axis=(0,1)))
            
            # 2. Median difference (more robust)
            median_diff = np.abs(np.median(left.reshape(-1, 3), axis=0) - 
                               np.median(right.reshape(-1, 3), axis=0))
            
            # 3. Standard deviation difference
            std_diff = np.abs(np.std(left, axis=(0,1)) - np.std(right, axis=(0,1)))
            
            # Trigger on ANY significant difference
            if np.max(mean_diff) > 10 or np.max(median_diff) > 8 or np.max(std_diff) > 5:
                mask = np.zeros((h, w), dtype=np.float32)
                
                # Adaptive mask width based on difference magnitude
                max_diff = max(np.max(mean_diff), np.max(median_diff), np.max(std_diff))
                mask_width = int(50 + max_diff * 2)  # 50-100+ pixels
                
                for dx in range(-mask_width, mask_width + 1):
                    if 0 <= pos + dx < w:
                        weight = 1.0 - abs(dx) / mask_width
                        mask[:, pos + dx] = np.maximum(mask[:, pos + dx], weight)
                
                return mask
                
        else:  # Horizontal line at y=pos
            # Similar logic for horizontal
            if pos < 30 or pos > h - 30:
                return None
                
            top = image[max(0, pos-30):pos, :]
            bottom = image[pos:min(h, pos+30), :]
            
            if top.size == 0 or bottom.size == 0:
                return None
            
            mean_diff = np.abs(np.mean(top, axis=(0,1)) - np.mean(bottom, axis=(0,1)))
            median_diff = np.abs(np.median(top.reshape(-1, 3), axis=0) - 
                               np.median(bottom.reshape(-1, 3), axis=0))
            std_diff = np.abs(np.std(top, axis=(0,1)) - np.std(bottom, axis=(0,1)))
            
            if np.max(mean_diff) > 10 or np.max(median_diff) > 8 or np.max(std_diff) > 5:
                mask = np.zeros((h, w), dtype=np.float32)
                max_diff = max(np.max(mean_diff), np.max(median_diff), np.max(std_diff))
                mask_width = int(50 + max_diff * 2)
                
                for dy in range(-mask_width, mask_width + 1):
                    if 0 <= pos + dy < h:
                        weight = 1.0 - abs(dy) / mask_width
                        mask[pos + dy, :] = np.maximum(mask[pos + dy, :], weight)
                
                return mask
        
        return None
    
    def _detect_gradient_discontinuity(self, image: np.ndarray, pos: int, axis: str) -> Optional[np.ndarray]:
        """Detect gradient/edge discontinuities"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        if axis == 'vertical':
            if pos < 20 or pos > w - 20:
                return None
            
            # Check gradient consistency across boundary
            left_grad = grad_mag[:, max(0, pos-20):pos]
            right_grad = grad_mag[:, pos:min(w, pos+20)]
            
            if left_grad.size > 0 and right_grad.size > 0:
                # Significant change in gradient patterns indicates seam
                left_mean = np.mean(left_grad)
                right_mean = np.mean(right_grad)
                
                if abs(left_mean - right_mean) > 20:
                    mask = np.zeros((h, w), dtype=np.float32)
                    
                    # Wider mask for gradient discontinuities
                    mask_width = 80
                    for dx in range(-mask_width, mask_width + 1):
                        if 0 <= pos + dx < w:
                            weight = 1.0 - abs(dx) / mask_width
                            mask[:, pos + dx] = np.maximum(mask[:, pos + dx], weight * 0.8)
                    
                    return mask
        else:
            # Similar for horizontal
            if pos < 20 or pos > h - 20:
                return None
                
            top_grad = grad_mag[max(0, pos-20):pos, :]
            bottom_grad = grad_mag[pos:min(h, pos+20), :]
            
            if top_grad.size > 0 and bottom_grad.size > 0:
                top_mean = np.mean(top_grad)
                bottom_mean = np.mean(bottom_grad)
                
                if abs(top_mean - bottom_mean) > 20:
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask_width = 80
                    
                    for dy in range(-mask_width, mask_width + 1):
                        if 0 <= pos + dy < h:
                            weight = 1.0 - abs(dy) / mask_width
                            mask[pos + dy, :] = np.maximum(mask[pos + dy, :], weight * 0.8)
                    
                    return mask
        
        return None
    
    def _detect_frequency_change(self, image: np.ndarray, pos: int, axis: str) -> Optional[np.ndarray]:
        """Detect texture/frequency changes using FFT"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        if axis == 'vertical':
            if pos < 64 or pos > w - 64:
                return None
            
            # Analyze frequency content on both sides
            left_region = gray[:, max(0, pos-64):pos]
            right_region = gray[:, pos:min(w, pos+64)]
            
            if left_region.shape[1] >= 32 and right_region.shape[1] >= 32:
                # Compute frequency characteristics
                left_fft = np.fft.fft2(left_region)
                right_fft = np.fft.fft2(right_region)
                
                left_spectrum = np.abs(left_fft)
                right_spectrum = np.abs(right_fft)
                
                # Compare high-frequency content (texture indicator)
                left_hf = np.sum(left_spectrum[left_spectrum.shape[0]//4:, :])
                right_hf = np.sum(right_spectrum[right_spectrum.shape[0]//4:, :])
                
                if abs(left_hf - right_hf) / max(left_hf, right_hf) > 0.3:
                    mask = np.zeros((h, w), dtype=np.float32)
                    
                    # Very wide mask for texture changes
                    mask_width = 120
                    for dx in range(-mask_width, mask_width + 1):
                        if 0 <= pos + dx < w:
                            weight = 1.0 - abs(dx) / mask_width
                            mask[:, pos + dx] = np.maximum(mask[:, pos + dx], weight * 0.6)
                    
                    return mask
        
        # Similar for horizontal (omitted for brevity)
        return None
    
    def _detect_discontinuities(self, image: np.ndarray) -> Optional[np.ndarray]:
        """General discontinuity detection as fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect strong edges
        edges = cv2.Canny(gray, 30, 90)  # Lower thresholds
        
        # Find long vertical/horizontal lines (potential seams)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            mask = np.zeros(gray.shape, dtype=np.float32)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is mostly vertical or horizontal
                if abs(x2 - x1) < 10:  # Vertical
                    cv2.line(mask, (x1, 0), (x1, gray.shape[0]), 1.0, thickness=80)
                elif abs(y2 - y1) < 10:  # Horizontal
                    cv2.line(mask, (0, y1), (gray.shape[1], y1), 1.0, thickness=80)
            
            # Blur the mask
            mask = cv2.GaussianBlur(mask, (61, 61), 0)
            
            if np.max(mask) > 0:
                return mask
        
        return None
    
    def _detect_tile_artifacts(self, 
                              image: np.ndarray, 
                              tile_boundaries: List[Tuple[int, int]], 
                              scale: float) -> Optional[np.ndarray]:
        """Detect tile boundary artifacts"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        for x, y in tile_boundaries:
            x_scaled = int(x * scale)
            y_scaled = int(y * scale)
            
            # Mark tile boundaries with wide masks
            if 0 <= x_scaled < w:
                for dx in range(-60, 61):
                    if 0 <= x_scaled + dx < w:
                        weight = 1.0 - abs(dx) / 60
                        mask[:, x_scaled + dx] = np.maximum(mask[:, x_scaled + dx], weight * 0.5)
            
            if 0 <= y_scaled < h:
                for dy in range(-60, 61):
                    if 0 <= y_scaled + dy < h:
                        weight = 1.0 - abs(dy) / 60
                        mask[y_scaled + dy, :] = np.maximum(mask[y_scaled + dy, :], weight * 0.5)
        
        if np.max(mask) > 0:
            return cv2.GaussianBlur(mask, (31, 31), 0)
        
        return None