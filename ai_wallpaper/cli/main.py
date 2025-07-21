#!/usr/bin/env python3
"""
Main CLI entry point for AI Wallpaper System
Unified command-line interface for all wallpaper generation
"""

import click
import sys
from pathlib import Path
from typing import Optional

from ..core import get_logger, get_config, handle_error
from ..core.exceptions import AIWallpaperError

# Configure Click
CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help'],
    max_content_width=120
)

@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.version_option(version='4.5.3', prog_name='AI Wallpaper Generator')
@click.option('--config', type=click.Path(), help='Custom config directory')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--dry-run', is_flag=True, help='Show plan without executing')
@click.pass_context
def cli(ctx, config, verbose, dry_run):
    """AI Wallpaper Generator - Create stunning 4K wallpapers with AI
    
    This tool generates ultra-high-quality desktop wallpapers using various AI models
    including FLUX, DALL-E 3, GPT-Image-1, and SDXL with automatic prompt generation
    and weather integration.
    """
    # Store options in context
    ctx.ensure_object(dict)
    ctx.obj['config_dir'] = config
    ctx.obj['verbose'] = verbose
    ctx.obj['dry_run'] = dry_run
    
    # If no command specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        
@cli.command()
@click.option('--prompt', help='Custom prompt (bypasses theme system)')
@click.option('--theme', help='Force specific theme')
@click.option('--model', type=click.Choice(['flux', 'dalle3', 'gpt-image-1', 'sdxl']), 
              help='Force specific model')
@click.option('--random-model', is_flag=True, help='Use weighted random model selection')
@click.option('--random-params', is_flag=True, help='Randomize valid parameters')
@click.option('--seed', type=int, help='Seed for reproducible generation')
@click.option('--no-upscale', is_flag=True, help='Skip upscaling stage')
@click.option('--no-wallpaper', is_flag=True, help='Generate only, don\'t set wallpaper')
@click.option('--save-stages', is_flag=True, help='Save intermediate stage images')
@click.option('--output', type=click.Path(), help='Custom output path')
@click.option('--resolution', type=str, 
              help='Target resolution as WIDTHxHEIGHT (e.g., 3840x2160) or preset name')
@click.option('--quality-mode', type=click.Choice(['fast', 'balanced', 'ultimate']),
              default='balanced', help='Quality mode - ultimate takes longer but maximizes quality')
@click.option('--no-tiled-refinement', is_flag=True,
              help='Disable tiled refinement pass (faster but lower quality)')
@click.pass_context
def generate(ctx, prompt, theme, model, random_model, random_params, seed, 
            no_upscale, no_wallpaper, save_stages, output, resolution, quality_mode, no_tiled_refinement):
    """Generate a new AI wallpaper
    
    This is the main command that generates a wallpaper using the selected model
    and options. By default, it will select a random theme, generate an appropriate
    prompt, create the image, upscale to 4K, and set as desktop wallpaper.
    """
    try:
        logger = get_logger()
        logger.info("=== AI WALLPAPER GENERATION STARTED ===")
        
        # Import here to avoid circular imports
        from ..commands.generate import GenerateCommand
        
        # Create command with options
        cmd = GenerateCommand(
            config_dir=ctx.obj.get('config_dir'),
            verbose=ctx.obj.get('verbose'),
            dry_run=ctx.obj.get('dry_run')
        )
        
        # Execute generation
        result = cmd.execute(
            prompt=prompt,
            theme=theme,
            model=model,
            random_model=random_model,
            random_params=random_params,
            seed=seed,
            no_upscale=no_upscale,
            no_wallpaper=no_wallpaper,
            save_stages=save_stages,
            output_path=output,
            resolution=resolution,
            quality_mode=quality_mode,
            no_tiled_refinement=no_tiled_refinement
        )
        
        # Display result
        if result:
            logger.info("=== GENERATION COMPLETE ===")
            logger.info(f"Wallpaper saved to: {result['image_path']}")
            logger.info(f"Total time: {result['duration']:.1f} seconds")
            
    except AIWallpaperError as e:
        handle_error(e, "Generation failed")
    except Exception as e:
        handle_error(e, "Unexpected error during generation")
        
@cli.command()
@click.option('--model', type=click.Choice(['flux', 'dalle3', 'gpt-image-1', 'sdxl']),
              help='Test specific model')
@click.option('--component', type=click.Choice(['prompt', 'image', 'wallpaper', 'theme']),
              help='Component to test')
@click.option('--quick', is_flag=True, help='Fast test mode')
@click.pass_context
def test(ctx, model, component, quick):
    """Test system components
    
    Run various tests to ensure the system is working correctly.
    This includes testing prompt generation, image generation, wallpaper setting,
    and theme selection.
    """
    try:
        logger = get_logger()
        logger.info("=== SYSTEM TEST STARTED ===")
        
        from ..commands.test import TestCommand
        
        cmd = TestCommand(
            config_dir=ctx.obj.get('config_dir'),
            verbose=ctx.obj.get('verbose')
        )
        
        result = cmd.execute(
            model=model,
            component=component,
            quick=quick
        )
        
        if result['success']:
            logger.info("=== ALL TESTS PASSED ===")
        else:
            logger.error("=== SOME TESTS FAILED ===")
            sys.exit(1)
            
    except AIWallpaperError as e:
        handle_error(e, "Test failed")
    except Exception as e:
        handle_error(e, "Unexpected error during testing")
        
@cli.command()
@click.option('--show', is_flag=True, help='Display current configuration')
@click.option('--validate', is_flag=True, help='Validate all config files')
@click.option('--set', 'set_value', help='Set configuration value (KEY=VALUE)')
@click.option('--reset', is_flag=True, help='Reset to defaults')
@click.pass_context
def config(ctx, show, validate, set_value, reset):
    """Manage configuration
    
    View, modify, and validate the AI Wallpaper configuration.
    Configuration files are stored in YAML format and control all aspects
    of the system including model settings, paths, and behavior.
    """
    try:
        logger = get_logger()
        
        from ..commands.config import ConfigCommand
        
        cmd = ConfigCommand(
            config_dir=ctx.obj.get('config_dir'),
            verbose=ctx.obj.get('verbose')
        )
        
        if show:
            cmd.show_config()
        elif validate:
            cmd.validate_config()
        elif set_value:
            key, value = set_value.split('=', 1)
            cmd.set_config(key, value)
        elif reset:
            if click.confirm("Reset all configuration to defaults?"):
                cmd.reset_config()
        else:
            click.echo("Specify an action: --show, --validate, --set, or --reset")
            
    except AIWallpaperError as e:
        handle_error(e, "Configuration operation failed")
    except Exception as e:
        handle_error(e, "Unexpected error in configuration")
        
@cli.command()
@click.option('--list', 'list_models', is_flag=True, help='List available models')
@click.option('--info', help='Show detailed info for specific model')
@click.option('--check', help='Check if model is ready to use')
@click.option('--install', help='Download/install specific model')
@click.pass_context
def models(ctx, list_models, info, check, install):
    """Manage AI models
    
    List available models, check their status, view detailed information,
    or install missing models. Each model has different capabilities and
    requirements.
    """
    try:
        logger = get_logger()
        
        from ..commands.models import ModelsCommand
        
        cmd = ModelsCommand(
            config_dir=ctx.obj.get('config_dir'),
            verbose=ctx.obj.get('verbose')
        )
        
        if list_models:
            cmd.list_models()
        elif info:
            cmd.show_model_info(info)
        elif check:
            cmd.check_model(check)
        elif install:
            if click.confirm(f"Install model '{install}'? This may take a while."):
                cmd.install_model(install)
        else:
            click.echo("Specify an action: --list, --info, --check, or --install")
            
    except AIWallpaperError as e:
        handle_error(e, "Model operation failed")
    except Exception as e:
        handle_error(e, "Unexpected error in model management")

def main():
    """Main entry point"""
    cli()

if __name__ == '__main__':
    main()