#!/usr/bin/env python3
"""
Smart Quality Refinement - Maximum Detail Preservation
Only refine what needs refinement
NO ERROR TOLERANCE - ALL OR NOTHING
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..core import get_logger
from .smart_detector import SmartArtifactDetector
from ..utils.lossless_save import save_lossless_png
from ..core.vram_calculator import VRAMCalculator

class SmartQualityRefiner:
    """Smart refinement that preserves detail - NO FALLBACKS"""
    
    def __init__(self, sdxl_model):
        self.model = sdxl_model
        self.logger = get_logger()
        self.detector = SmartArtifactDetector()
        self.vram_calc = VRAMCalculator()
        
        # Get config settings - MUST EXIST
        self.config = self.model.config['pipeline']['stages']['initial_refinement']
        if not self.config:
            raise RuntimeError("initial_refinement config MISSING - CANNOT PROCEED")
        
    def refine_smart(self,
                    image_path: Path,
                    prompt: str,
                    seed: int,
                    params: Dict[str, Any],
                    temp_files: List[Path]) -> Dict[str, Any]:
        """
        Smart refinement - only as much as needed.
        NO ERROR TOLERANCE - FAILS COMPLETELY ON ANY ERROR
        """
        metadata = getattr(self.model, 'generation_metadata', {})
        if not metadata:
            raise RuntimeError("generation_metadata NOT INITIALIZED - CRITICAL ERROR")
        
        # Quick analysis
        self.logger.info("Analyzing image for quality issues...")
        analysis = self.detector.quick_analysis(image_path, metadata)
        
        # Load image
        current_image = Image.open(image_path)
        current_path = image_path
        w, h = current_image.size
        
        passes_done = 0
        
        # Check if multi-pass is enabled and needed
        multi_pass_enabled = self.config.get('multi_pass_enabled', False)
        
        # Decide strategy based on analysis
        if not multi_pass_enabled or not analysis['needs_multipass']:
            # Single light refinement pass for all images
            self.logger.info("Single quality enhancement pass")
            
            refined = self._single_quality_pass(
                current_image, current_path, prompt, seed, temp_files, params
            )
            
            return {
                'image': refined['image'],
                'image_path': refined['path'],
                'size': refined['image'].size,
                'method': 'single_pass',
                'passes': 1
            }
        
        else:
            # Multi-pass for specific issues
            self.logger.info("Issues detected - AGGRESSIVE multi-pass refinement")
            
            # Determine severity and adjust strategy
            severity = analysis.get('severity', 'high')
            seam_count = analysis.get('seam_count', 0)
            
            # Pass 1: Initial coherence (slightly stronger for seams)
            self.logger.info("Pass 1: Initial coherence pass")
            pass1_config = {
                'denoising_strength': 0.12 if severity == 'critical' else self.config['coherence_strength'],
                'steps': self.config['coherence_steps'],
                'guidance_scale': 4.5,  # Lower guidance for better blending
                'prompt_suffix': ", perfectly seamless and coherent, unified lighting"
            }
            pass1 = self._execute_pass(
                current_image, current_path, prompt, seed + 1000,
                pass1_config, "coherence", temp_files, params
            )
            current_image = pass1['image']
            current_path = pass1['path']
            passes_done += 1
            
            # Pass 2: Targeted fixes with STRONGER settings
            if analysis.get('mask') is not None:
                self.logger.info("Pass 2: AGGRESSIVE targeted artifact removal")
                
                # Create inpaint pipeline if needed
                if not hasattr(self.model, 'inpaint_pipe') or self.model.inpaint_pipe is None:
                    self.model._create_inpaint_pipeline()
                
                if self.model.inpaint_pipe is None:
                    raise RuntimeError("INPAINT PIPELINE CREATION FAILED - CANNOT PROCEED")
                
                # Stronger settings for critical seams
                pass2_config = {
                    'denoising_strength': 0.35 if severity == 'critical' else self.config['targeted_strength'],
                    'steps': 100 if severity == 'critical' else self.config['targeted_steps'],
                    'guidance_scale': 5.5  # Lower for better blending
                }
                pass2 = self._execute_masked_pass(
                    current_image, current_path, analysis['mask'],
                    prompt, seed + 2000, pass2_config, "targeted", temp_files
                )
                current_image = pass2['image']
                current_path = pass2['path']
                passes_done += 1
                
                # Pass 2.5: Second targeted pass if still critical
                if severity == 'critical' and seam_count > 2:
                    self.logger.info("Pass 2.5: Second targeted pass for persistent seams")
                    
                    # Re-analyze after first pass
                    reanalysis = self.detector.quick_analysis(current_path, metadata)
                    
                    if reanalysis.get('mask') is not None:
                        pass2_5_config = {
                            'denoising_strength': 0.28,
                            'steps': 80,
                            'guidance_scale': 6.0
                        }
                        pass2_5 = self._execute_masked_pass(
                            current_image, current_path, reanalysis['mask'],
                            prompt, seed + 2500, pass2_5_config, "targeted2", temp_files
                        )
                        current_image = pass2_5['image']
                        current_path = pass2_5['path']
                        passes_done += 1
            
            # Pass 3: Detail enhancement
            self.logger.info("Pass 3: Detail enhancement")
            pass3_config = {
                'denoising_strength': self.config['detail_strength'],
                'steps': self.config['detail_steps'],
                'guidance_scale': 7.5,
                'prompt_suffix': ", sharp detailed textures, high quality"
            }
            pass3 = self._execute_pass(
                current_image, current_path, prompt, seed + 3000,
                pass3_config, "detail", temp_files, params
            )
            current_image = pass3['image']
            current_path = pass3['path']
            passes_done += 1
            
            # Pass 4: FINAL UNIFICATION PASS (new!)
            if severity == 'critical' or seam_count > 0:
                self.logger.info("Pass 4: Final unification pass")
                pass4_config = {
                    'denoising_strength': 0.03,  # VERY light
                    'steps': 40,
                    'guidance_scale': 7.0,
                    'prompt_suffix': ", perfectly unified and seamless"
                }
                pass4 = self._execute_pass(
                    current_image, current_path, prompt, seed + 4000,
                    pass4_config, "unify", temp_files
                )
                current_image = pass4['image']
                current_path = pass4['path']
                passes_done += 1
            
            return {
                'image': current_image,
                'image_path': current_path,
                'size': current_image.size,
                'method': 'multi_pass',
                'passes': passes_done
            }
    
    def _single_quality_pass(self, image, image_path, prompt, seed, temp_files, params=None):
        """Single quality enhancement pass"""
        # Use default refinement settings - MUST EXIST
        pass_config = {
            'denoising_strength': self.config['denoising_strength'],  # MUST EXIST
            'steps': self.config['steps'],                          # MUST EXIST
            'guidance_scale': 6.0
        }
        
        return self._execute_pass(
            image, image_path, prompt, seed,
            pass_config, "quality", temp_files, params
        )
    
    def _execute_pass(self, image, image_path, prompt, seed, config, pass_name, temp_files, params=None):
        """Execute a refinement pass - NO ERROR TOLERANCE"""
        # Get parameters
        strength = config['denoising_strength']  # MUST EXIST
        steps = config['steps']                  # MUST EXIST
        guidance = config.get('guidance_scale', 6.0)
        prompt_suffix = config.get('prompt_suffix', '')
        
        enhanced_prompt = prompt + prompt_suffix
        
        self.logger.info(f"Executing {pass_name} pass: strength={strength}, steps={steps}")
        
        # Check VRAM and tiled refinement preference
        w, h = image.size
        strategy = self.vram_calc.determine_refinement_strategy(w, h)
        
        # If no_tiled_refinement is True, force full refinement
        no_tiled = params and params.get('no_tiled_refinement', False)
        
        if strategy['strategy'] == 'full' or no_tiled:
            # Full refinement
            generator = torch.Generator(device=self.model.device).manual_seed(seed)
            refined = self.model.refiner_pipe(
                prompt=enhanced_prompt,
                image=image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        else:
            # Tiled refinement
            from .tiled_refiner import TiledRefiner
            tiled = TiledRefiner(
                pipeline=self.model.refiner_pipe,
                vram_calculator=self.vram_calc
            )
            
            # Save temp image for tiled processing
            temp_path = image_path.parent / f"temp_{pass_name}.png"
            save_lossless_png(image, temp_path)
            temp_files.append(temp_path)
            
            # Use correct method name
            result_path = tiled.refine_tiled(
                image_path=temp_path,
                prompt=enhanced_prompt,
                base_strength=strength,
                base_steps=steps,
                seed=seed
            )
            refined = Image.open(result_path)
            temp_files.append(result_path)
        
        # Save result
        result_path = image_path.parent / f"{pass_name}_refined.png"
        save_lossless_png(refined, result_path)
        temp_files.append(result_path)
        
        return {'image': refined, 'path': result_path}
    
    def _execute_masked_pass(self, image, image_path, mask, prompt, seed, config, pass_name, temp_files):
        """Execute masked inpainting pass - NO ERROR TOLERANCE"""
        # Validate mask
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"INVALID MASK TYPE: {type(mask)} - MUST BE numpy.ndarray")
        
        # Convert mask to PIL
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Aggressive dilation for better blending
        mask_array = np.array(mask_pil)
        
        # Multiple dilation passes for critical seams
        if self.model.generation_metadata.get('seam_details'):
            max_expansion = max(
                d.get('expansion_size', 100) 
                for d in self.model.generation_metadata['seam_details']
            )
            
            # Larger kernel for bigger expansions
            kernel_size = min(31, max(15, int(max_expansion * 0.05)))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_array = cv2.dilate(mask_array, kernel, iterations=2)  # Two iterations
        else:
            # Standard dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        
        # Stronger blur for smoother transitions
        blur_size = 41  # Larger blur
        mask_array = cv2.GaussianBlur(mask_array, (blur_size, blur_size), 0)
        mask_pil = Image.fromarray(mask_array)
        
        # Parameters - MUST EXIST
        strength = config['denoising_strength']  # MUST EXIST
        steps = config['steps']                  # MUST EXIST
        guidance = config.get('guidance_scale', 6.5)
        
        # Inpaint - NO ERROR CATCHING
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        w, h = image.size
        
        refined = self.model.inpaint_pipe(
            prompt=prompt + ", seamless perfect quality",
            image=image,
            mask_image=mask_pil,
            width=w,
            height=h,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]
        
        # Save result
        result_path = image_path.parent / f"{pass_name}_refined.png"
        save_lossless_png(refined, result_path)
        temp_files.append(result_path)
        
        return {'image': refined, 'path': result_path}