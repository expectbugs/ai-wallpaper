Program Refactor:

1: Make it way more modular instead of a giant monolithic script.  Each function should be its own script or series of scripts, however makes the most sense.

2: All the hardcoded paths values and constants and everything will be exported to a config file or files instead.

3: The commandline arguments and features will be upgraded to include more AI models and generation strategies.

4: A script will be created for using DALL-E the way the daily_wallpaper_dalle.py script uses it to generate a 1792x1024 image to be upscaled to 4k.

5: Another script will be created for using the newer and improved chatgpt image gen that works like the daily_wallpaper_gpt.py script, and another that is like the daily_wallpaper_gpt2.py script (one of them invokes gpt-image-1 directly with deepseek's prompt, the other prompts gpt-4o with the themes and context and asks it to return an image).

6: Yet another script will be created for using the latest SDXL, with options for LoRas and more to pimp out the image extra (maybe that can be selected by DeepSeek or random theme selector even!)

7: Core scripts for more useful imagegen models.

8: Config options to enable/disable or change default model, schedulers, upscaling, img2img refinement, prompt engineering LLM, and many many other options.  "random" is a valid value for most options that selects an option at random (but ONLY options that work with the current other settings, no invalid combinations can be randomized).

9: Command-line argument to select model.

10: Verification that each model's script changes everything needed for that model, including prompt to deepseek and scheduler and everything optimized for the model at its absolute highest quality no matter how long it takes or how much space it takes up.